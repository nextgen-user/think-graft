"""Render the article's share card from its current headline."""
import asyncio, html, os, pathlib
from playwright.async_api import async_playwright
ROOT=pathlib.Path(__file__).resolve().parents[1]
async def main():
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path=os.environ.get('CHROMIUM'),args=['--disable-gpu','--no-sandbox'])
        page=await browser.new_page(viewport={'width':1200,'height':630},device_scale_factor=1)
        await page.goto('http://127.0.0.1:8765/2/',wait_until='domcontentloaded')
        title=html.escape(await page.locator('h1').inner_text())
        await page.set_content('''<html><style>*{box-sizing:border-box}body{margin:0;background:#f8f7f3;color:#20221f;padding:50px 65px;font:22px system-ui}.top{color:#285b46;font-size:18px;letter-spacing:.04em}h1{font:60px/1.12 Georgia;letter-spacing:-.025em;max-width:1060px;margin:30px 0 26px}.sub{font:25px/1.45 Georgia;max-width:1000px;color:#51574e}.bottom{border-top:2px solid #285b46;margin-top:35px;padding-top:23px;display:flex;gap:48px;font-size:19px}.bottom b{display:block;font:38px Georgia;color:#285b46;margin-bottom:8px}.foot{position:absolute;bottom:24px;font-size:14px;color:#656b62}</style><body><div class="top">PROBE GRAFT / EXPERIMENT 02</div><h1>'''+title+'''</h1><div class="sub">Three output-row edits in Qwen. A complete MATH-500 evaluation.<br>Recorded reasoning, reproducible weights, and the loops that remain.</div><div class="bottom"><div><b>55.2%</b>276 / 500 correct</div><div><b>3</b>output rows edited</div><div><b>0</b>internal tensors changed</div></div><div class="foot">Vaibhav · probe-graft.vercel.app/2/</div></body></html>''')
        await page.screenshot(path=str(ROOT/'blog/2/preview.png'))
        await browser.close()
asyncio.run(main())
