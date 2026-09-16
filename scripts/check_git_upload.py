"""Reject private data and credential patterns in every commit being uploaded.

Only reports paths and finding categories, never matching secret values.
"""
from __future__ import annotations
import argparse
import os
from pathlib import PurePosixPath
import re
import subprocess
import sys

PRIVATE_DIRS = {
    '.codex', '.agents', '.agent', '.github-local', 'data', 'models',
    'contracts', 'provenance', 'docs', 'skills', 'runs', 'outputs',
    'artifacts', 'node_modules', 'dist', 'build', '__pycache__',
    '.pytest_cache', '.pytest_tmp', '.venv', '.idea', '.vscode', '.playwright-cli',
}
PRIVATE_ROOT_FILES = {'MANIFEST.sha256', '文件清单.md'}
PRIVATE_EXTENSIONS = {
    '.xls', '.xlsx', '.xlsm', '.xlsb', '.csv', '.tsv', '.parquet',
    '.feather', '.arrow', '.h5', '.hdf5', '.npy', '.npz', '.joblib',
    '.pkl', '.pickle', '.pt', '.pth', '.onnx', '.db', '.sqlite', '.sqlite3',
    '.doc', '.docx', '.pdf', '.ppt', '.pptx', '.zip', '.7z', '.rar',
    '.tar', '.gz', '.bundle', '.pem', '.key', '.p12', '.pfx', '.kdbx',
    '.log', '.pyc', '.tsbuildinfo', '.dpapi',
}
RULES = {
    'private-key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----'),
    'provider-token': re.compile(r'(?<![A-Za-z0-9])(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16})'),
    'personal-absolute-path': re.compile(r'(?i)(?:[A-Z]:[/\\](?:Users|postgraduate1)|' + '/' + r'home/|' + '/' + r'Users/)'),
    'embedded-auth-url': re.compile(r'https?://[^\s/@:]+:[^\s/@]+@'),
    'assigned-credential': re.compile(r'(?i)(?:api[_-]?key|access[_-]?token|password|client[_-]?secret)[\"\x27]?\s*[:=]\s*[\"\x27]([^\"\x27\n]{8,})[\"\x27]'),
}
# Precisely scoped synthetic fixtures reviewed during repository preparation.
SYNTHETIC = {
    'tests/v1/test_provider_security.py': {'sk-' + 'abcdefghijklmnopqrstuvwxyz123456', 'local-test-value'},
    'tests/mvp/test_workbench.py': {'sk-' + 'test-synthetic-not-real'},
    'tests/mvp/test_diagnostic_agent.py': {'test-local-key'},
}

def git(*args):
    proc = subprocess.run(['git', *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode:
        raise RuntimeError('Git inspection failed; upload blocked.')
    return proc.stdout

def path_reason(path):
    parsed = PurePosixPath(path)
    parts = parsed.parts
    if (path in PRIVATE_ROOT_FILES or (parts and parts[0] in PRIVATE_DIRS)
        or path.startswith('configs/contracts/')
        or any(p in {'node_modules', '__pycache__', '.pytest_cache'} or p.startswith(('.venv', '.pytest_tmp')) for p in parts)
        or parsed.suffix.lower() in PRIVATE_EXTENSIONS
        or (parsed.name.lower().startswith('.env') and path != '.env.example')
        or parsed.name.lower().startswith('credentials')
        or 'service-account' in parsed.name.lower()):
        return 'private-path'
    return None

def inspect_tree(tree, seen):
    issues = []
    for record in git('ls-tree', '-rz', '--full-tree', tree).split(b'\0'):
        if not record:
            continue
        metadata, raw_path = record.split(b'\t', 1)
        mode, kind, oid = metadata.decode().split()
        path = raw_path.decode('utf-8')
        reason = path_reason(path)
        if reason:
            issues.append((path, reason))
            continue
        if kind != 'blob' or mode == '120000':
            issues.append((path, 'unreviewed-submodule-or-symlink'))
            continue
        identity = (path, oid)
        if identity in seen:
            continue
        seen.add(identity)
        data = git('cat-file', 'blob', oid)
        try:
            content = data.decode('utf-8-sig')
        except UnicodeDecodeError:
            issues.append((path, 'unreviewed-binary'))
            continue
        for sample in SYNTHETIC.get(path, set()):
            content = content.replace(sample, '')
        for label, rule in RULES.items():
            if rule.search(content):
                issues.append((path, label))
    return issues

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--ref')
    mode.add_argument('--tree')
    mode.add_argument('--pre-push', action='store_true')
    parser.add_argument('remote', nargs='*')
    args = parser.parse_args()
    roots = [args.ref] if args.ref else []
    if args.pre_push:
        for line in sys.stdin:
            local_ref, local_oid, remote_ref, remote_oid = line.split()
            if set(local_oid) == {'0'}:
                continue
            if not local_ref.startswith('refs/heads/') or not remote_ref.startswith('refs/heads/'):
                raise RuntimeError('Only reviewed branch refs may be pushed; backup refs and tags are blocked.')
            roots.append(local_oid)
    trees = {args.tree} if args.tree else set()
    for ref in roots:
        trees.update(git('log', '--format=%T', ref).decode().splitlines())
    seen = set()
    issues = set()
    for tree in trees:
        issues.update(inspect_tree(tree, seen))
    if issues:
        for path, reason in sorted(issues):
            print(f'BLOCKED {path}: {reason}', file=sys.stderr)
        return 1
    print(f'Upload audit passed: {len(trees)} tree(s), {len(seen)} text file version(s).')
    return 0

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, UnicodeError) as exc:
        print(f'Upload audit failed: {exc}', file=sys.stderr)
        raise SystemExit(1)
