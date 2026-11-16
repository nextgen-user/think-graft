"""Browser checks for the static article. Does not contact model servers."""
import asyncio,json,os,pathlib
from playwright.async_api import async_playwright
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=pathlib.Path('/tmp/probe-graft-qa');OUT.mkdir(exist_ok=True)
BASE='http://127.0.0.1:8765'
async def main():
    checks=[]
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path=os.environ.get('CHROMIUM'),headless=True,args=['--disable-gpu','--no-sandbox'])
        for label,width,height,scheme in [('desktop',1440,1000,'light'),('mobile',390,844,'light'),('dark',1280,900,'dark')]:
            context=await browser.new_context(viewport=dict(width=width,height=height),device_scale_factor=1,color_scheme=scheme)
            page=await context.new_page();errors=[]
            page.on('pageerror',lambda error:errors.append(str(error)))
            response=await page.goto(BASE+'/2/',wait_until='networkidle');assert response.status==200
            await page.locator('#play').wait_for(state='visible');await page.wait_for_function('document.querySelector("#play").disabled === false')
            assert await page.locator('h1').inner_text()=='Reasoning loops,\nthree output rows.'
            assert await page.evaluate('document.documentElement.scrollWidth<=innerWidth')
            await page.locator('#play').click();await page.wait_for_timeout(500)
            assert await page.locator('#ours-text').inner_text()
            await page.locator('#play').click();await page.locator('#ending').click()
            assert await page.locator('#ours-status').inner_text()=='RIGHT'
            await page.locator('#trace-choice').select_option('2')
            await page.wait_for_function('document.querySelector("#case-id").textContent.includes("1994")')
            await page.locator('#ending').click()
            assert await page.locator('#ours-status').inner_text()=='WRONG'
            assert await page.locator('#reference-status').inner_text()=='RIGHT'
            assert await page.evaluate('Number(document.querySelector("#position").value)>30000')
            await page.locator('#reset').click()
            assert await page.locator('#position').input_value()=='0'
            assert await page.locator('#ours-text').inner_text()=='Recording paused at the start.'
            await page.locator('#trace-choice').select_option('0');await page.wait_for_function('document.querySelector("#case-id").textContent.includes("572")')
            await page.screenshot(path=str(OUT/f'{label}.png'),full_page=True)
            math=await page.locator('.katex').count()
            checks.append(dict(view=label,status=response.status,horizontal_overflow=False,math_elements=math,errors=errors))
            assert not errors,errors
            if label=='desktop':
                for href in await page.locator('a[href]').evaluate_all('(a)=>a.map(e=>e.getAttribute("href"))'):
                    if href.startswith('#'):assert await page.locator(href).count(),href
                    elif href.startswith('/') or href.startswith('data/'):
                        url=BASE+href if href.startswith('/') else BASE+'/2/'+href
                        r=await context.request.get(url);assert r.ok,(url,r.status)
            await context.close()
        await browser.close()
    (OUT/'checks.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks,indent=2))
asyncio.run(main())
