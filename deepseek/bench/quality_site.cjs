// Functional + visual check of the bench's site outputs (Nov 14): what the 10-point static checklist cannot see.
// For each TAG: ~/deepseek-work/bench/TAG-site/index.html in headless Chromium (playwright-core from ~/nanobots).
// Screenshots (desktop 1280, phone 375, dark mode) -> ~/deepseek-work/bench/quality/TAG-*.png; results -> quality/TAG-site.json
// usage: node quality_site.cjs TAG [TAG..]
const { chromium } = require('~/nanobots/node_modules/playwright-core');
const fs = require('fs'), path = require('path'), os = require('os');
const B = path.join(os.homedir(), 'deepseek-work/bench'), Q = path.join(B, 'quality');
async function one(browser, tag) {
  const file = 'file://' + path.join(B, tag + '-site', 'index.html'); const r = { tag, errors: [] };
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 } }); const pg = await ctx.newPage();
  pg.on('pageerror', e => r.errors.push('pageerror: ' + e.message.slice(0, 160)));
  pg.on('console', m => { if (m.type() === 'error') r.errors.push('console: ' + m.text().slice(0, 160)); });
  await pg.goto(file); await pg.waitForTimeout(400);
  await pg.screenshot({ path: path.join(Q, tag + '-desktop.png'), fullPage: true });
  r.static = await pg.evaluate(() => {
    const txt = document.body.innerText; const prices = (txt.match(/\$\s?\d+(\.\d{2})?/g) || []).length;
    const inputs = [...document.querySelectorAll('input, textarea')];
    const labelled = inputs.filter(i => (i.id && document.querySelector(`label[for="${i.id}"]`)) || i.closest('label') || i.getAttribute('aria-label')).length;
    const anchors = [...document.querySelectorAll('a[href^="#"]')].map(a => a.getAttribute('href')).filter(h => h.length > 1);
    return { lang: document.documentElement.lang || null, viewport_meta: !!document.querySelector('meta[name=viewport]'),
             title: document.title, prices, words: txt.split(/\s+/).length, headings: [...document.querySelectorAll('h1,h2,h3')].map(h => h.tagName + ':' + h.innerText.slice(0, 30)),
             inputs: inputs.length, inputs_labelled: labelled, nav_anchors: anchors.length, nav_anchors_ok: anchors.filter(h => document.querySelector(h)).length,
             semantic: ['header', 'nav', 'main', 'section', 'footer'].filter(t => document.querySelector(t)), html_kb: Math.round(document.documentElement.outerHTML.length / 1024) };
  });
  // dark mode: the toggle button, its effect, and persistence across a reload
  const tog = await pg.$('button:has-text("dark"), button:has-text("Dark"), [aria-label*="ark" i], #theme-toggle, .theme-toggle, button[id*="dark" i], button[class*="dark" i], button[class*="theme" i]');
  r.dark = { found: !!tog };
  if (tog) {
    const before = await pg.evaluate(() => document.body.className + '|' + document.documentElement.className + '|' + getComputedStyle(document.body).backgroundColor);
    await tog.click(); await pg.waitForTimeout(400);
    const after = await pg.evaluate(() => document.body.className + '|' + document.documentElement.className + '|' + getComputedStyle(document.body).backgroundColor);
    r.dark.changes = before !== after; r.dark.storage = await pg.evaluate(() => JSON.stringify(localStorage));
    await pg.screenshot({ path: path.join(Q, tag + '-dark.png'), fullPage: false });
    await pg.reload(); await pg.waitForTimeout(400);
    const again = await pg.evaluate(() => document.body.className + '|' + document.documentElement.className + '|' + getComputedStyle(document.body).backgroundColor);
    r.dark.persists = again === after;
    await pg.evaluate(() => localStorage.clear()); await pg.reload(); await pg.waitForTimeout(300);
  }
  // form: an invalid email must show an inline error and not submit
  const email = await pg.$('input[type=email], input[name*=email i], input[id*=email i]');
  r.form = { email_field: !!email };
  if (email) {
    const url0 = pg.url();
    const fill = async (sel, v) => { const e = await pg.$(sel); if (e) await e.fill(v); };
    await fill('input[name*=name i]:not([type=email]), input[id*=name i]:not([type=email])', 'Test Person'); await email.fill('not-an-email');
    await fill('textarea', 'Hello there, a test message.');
    const textBefore = await pg.evaluate(() => document.body.innerText);
    const sub = await pg.$('form button[type=submit], form input[type=submit], form button:not([type=button])');
    if (sub) await sub.click(); else await email.press('Enter');
    await pg.waitForTimeout(500);
    const textAfter = await pg.evaluate(() => document.body.innerText);
    r.form.stayed = pg.url() === url0;
    r.form.inline_error = textAfter.length > textBefore.length && /email|valid|address/i.test(textAfter.replace(textBefore, ''))
                          || await pg.evaluate(() => [...document.querySelectorAll('[role=alert], .error, .error-message, [class*=error], [id*=error]')].some(e => e.offsetParent && e.innerText.trim().length > 3));
    r.form.native_invalid = await pg.evaluate(() => { const e = document.querySelector('input[type=email]'); return e ? !e.checkValidity() : null; });
    // a valid submission: what happens (a confirmation is good, a navigation away is not)
    await email.fill('test@example.com'); if (sub) await sub.click(); else await email.press('Enter'); await pg.waitForTimeout(600);
    const textOk = await pg.evaluate(() => document.body.innerText);
    r.form.valid_stays = pg.url() === url0; r.form.valid_feedback = /thank|sent|received|success|soon/i.test(textOk) && !/thank|sent|received|success/i.test(textAfter);
  }
  // phone width: one column, no horizontal scroll
  await pg.setViewportSize({ width: 375, height: 800 }); await pg.reload(); await pg.waitForTimeout(400);
  await pg.screenshot({ path: path.join(Q, tag + '-mobile.png'), fullPage: true });
  r.mobile = await pg.evaluate(() => {
    const sw = document.documentElement.scrollWidth; const cards = [...document.querySelectorAll('*')].filter(e => /\$\s?\d/.test(e.innerText || '') && e.children.length > 0 && e.innerText.length < 300);
    const xs = new Set(cards.map(e => Math.round(e.getBoundingClientRect().left)));
    return { scroll_width: sw, no_hscroll: sw <= 376, menu_columns_seen: xs.size };
  });
  await ctx.close(); fs.writeFileSync(path.join(Q, tag + '-site.json'), JSON.stringify(r, null, 1)); return r;
}
(async () => {
  const browser = await chromium.launch({ headless: true });
  for (const tag of process.argv.slice(2)) {
    try { const r = await one(browser, tag); console.log(JSON.stringify({ tag, errors: r.errors.length, dark: r.dark, form: r.form, mobile: r.mobile, static: { prices: r.static.prices, words: r.static.words, labelled: `${r.static.inputs_labelled}/${r.static.inputs}`, nav: `${r.static.nav_anchors_ok}/${r.static.nav_anchors}`, semantic: r.static.semantic.join(','), kb: r.static.html_kb, lang: r.static.lang } })); }
    catch (e) { console.log(JSON.stringify({ tag, crash: e.message.slice(0, 200) })); }
  }
  await browser.close();
})();
