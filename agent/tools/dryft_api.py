"""Read-only Dryft API helper. Loads DRYFT_TOKEN/DRYFT_API from the ignored .env; never prints them."""
import json, os, sys, urllib.request, urllib.error
env = {}
for line in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '.env')):
    line = line.strip()
    if '=' in line and not line.startswith('#'):
        k, v = line.split('=', 1); env[k.strip().removeprefix('export ').strip()] = v.strip().strip('"\'')
BASE = env.get('DRYFT_API', 'https://htn.dryft.ai').rstrip('/')
def get(path):
    req = urllib.request.Request(BASE + path, headers={'Authorization': 'Bearer ' + env['DRYFT_TOKEN'], 'Accept': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read() or b'{}')
    except urllib.error.HTTPError as e:
        return {'_http_error': e.code, 'body': e.read()[:500].decode('utf8', 'replace')}
if __name__ == '__main__':
    print(json.dumps(get(sys.argv[1]), indent=1)[: int(sys.argv[2]) if len(sys.argv) > 2 else 20000])
