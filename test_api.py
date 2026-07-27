"""
End-to-end test of the Hunyuan3D-2 API server.
Usage: python test_api.py
"""
import base64
import json
import urllib.request
import urllib.error
import sys
import os

SERVER_URL = 'http://localhost:8081'
DEMO_IMAGE = 'assets/demo.png'
OUTPUT_GLB = 'gradio_cache/test_output.glb'


def test_generate():
    """Test POST /generate with a real image."""
    # Read and encode the demo image
    with open(DEMO_IMAGE, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')

    url = f'{SERVER_URL}/generate'
    data = json.dumps({'image': img_b64}).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})

    print(f'[POST] {url}')
    print(f'  Image: {DEMO_IMAGE} ({len(img_b64)} base64 chars)')
    print(f'  Waiting for shape generation (30-60s)...')
    sys.stdout.flush()

    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            glb_data = resp.read()
            with open(OUTPUT_GLB, 'wb') as f:
                f.write(glb_data)
            print(f'\n  ✅ SUCCESS!')
            print(f'  Status: {resp.status}')
            print(f'  Size: {len(glb_data)} bytes')
            print(f'  Saved to: {OUTPUT_GLB}')
            return True
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:500]
        print(f'\n  ❌ HTTP Error {e.code}: {body}')
        return False
    except Exception as e:
        print(f'\n  ❌ Error: {e}')
        return False


def test_status():
    """Test GET /status endpoint."""
    url = f'{SERVER_URL}/status/test123'
    print(f'\n[GET] {url}')
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            body = json.loads(resp.read().decode())
            print(f'  Status: {resp.status}')
            print(f'  Body: {body}')
            return True
    except Exception as e:
        print(f'  ❌ Error: {e}')
        return False


if __name__ == '__main__':
    print('=' * 60)
    print('Hunyuan3D-2 API Server End-to-End Test')
    print('=' * 60)

    # First verify server is alive
    alive = test_status()
    if not alive:
        print('\n⚠️  Server not responding. Is it running on port 8081?')
        sys.exit(1)

    # Then test actual generation
    success = test_generate()

    print('\n' + '=' * 60)
    if success:
        print('🎉 End-to-end test PASSED')
    else:
        print('❌ End-to-end test FAILED')
    print('=' * 60)
