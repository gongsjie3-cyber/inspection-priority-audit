"""Package the reviewed, self-contained report for GitHub Pages (stdlib only)."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = 'https://github.com/gongsjie3-cyber/inspection-priority-audit'
SITE_URL = 'https://gongsjie3-cyber.github.io/inspection-priority-audit/'


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Report template changed; review the Pages wrapper: {old}')
    return text.replace(old, new, 1)


def site_files():
    source = (ROOT / '项目展示.html').read_bytes()
    pdf = (ROOT / '项目报告.pdf').read_bytes()
    # Publishing is restricted to the artifacts recorded in the latest review.
    review = json.loads((ROOT / 'results/verification.json').read_text(encoding='utf-8'))
    for name, content in [('项目展示.html', source), ('项目报告.pdf', pdf)]:
        if hashlib.sha256(content).hexdigest() != review['artifact_sha256'][name]:
            raise ValueError(f'{name} changed; review it and refresh verification.json before publishing.')
    page = source.decode('utf-8')
    page = replace_once(page, '<style>',
        '<meta name="description" content="用公开餐饮巡检记录，比较风险排序、经验规则与地区覆盖配额。可交互查看预算情景、历史候选及完整研究报告。">\n'
        f'<link rel="canonical" href="{SITE_URL}">\n<style>')
    page = replace_once(page, '</style>',
        '.hero-actions{display:flex;flex-wrap:wrap;gap:12px;margin-top:24px}'
        '.hero-actions a{display:inline-block;padding:10px 17px;border:1px solid #aad9d1;'
        'border-radius:5px;color:#fff;text-decoration:none;font-size:14px}'
        '.hero-actions a:first-child{background:#e5f2ed;color:#17333c;border-color:#e5f2ed}'
        '.hero-actions a:hover{background:#2b555a;color:#fff}'
        '.hero-actions a:focus-visible{outline:3px solid #f2bb78;outline-offset:3px}'
        '</style>')
    page = replace_once(page, '</div></header>',
        '<div class="hero-actions"><a href="./report.pdf">阅读五页研究报告（PDF）</a>'
        f'<a href="{REPOSITORY}">查看源码与核验记录</a></div></div></header>')
    page = replace_once(page, '<main><div class="wrap">',
        '<main><div class="wrap"><noscript><p class="caveat">互动表格需要开启 JavaScript。'
        '也可直接<a href="./report.pdf">阅读 PDF 报告</a>查看完整结论。</p></noscript>')
    return {'index.html': page.encode('utf-8'), 'report.pdf': pdf, '.nojekyll': b''}


def main(check=False):
    out = ROOT / 'docs'
    expected = site_files()
    if check:
        stale = [name for name, content in expected.items()
                 if not (out / name).exists() or (out / name).read_bytes() != content]
        if stale:
            raise SystemExit('Pages files are missing or stale: ' + ', '.join(stale))
        print('Pages files match the reviewed report and current wrapper.')
        return
    out.mkdir(exist_ok=True)
    for name, content in expected.items():
        (out / name).write_bytes(content)
    print(f'Prepared {len(expected)} static files in docs/ for {SITE_URL}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true', help='Verify the published files without writing.')
    main(parser.parse_args().check)
