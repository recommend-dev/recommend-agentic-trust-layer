// Real check → open the comparison modal → screenshot the evidence-weight panel.
const puppeteer = require('puppeteer-core');
(async () => {
  const b = await puppeteer.launch({
    executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    headless: 'new', args: ['--no-sandbox'] });
  const p = await b.newPage();
  p.on('pageerror', e => console.log('PAGE ERROR:', e.message));
  await p.goto('http://localhost:8899', { waitUntil: 'networkidle0' });
  await p.type('#claim', 'The Great Wall of China is visible from space with the naked eye');
  await p.$eval('#deep', el => { el.checked = false; });
  await p.click('#go');
  await p.waitForFunction(() => lastVerdict && !lastVerdict.preliminary, { timeout: 120000 });
  const v = await p.evaluate(() => lastVerdict);
  console.log(`ours: ${v.label} score=${v.score} conf=${v.confidence} sources=${v.n_sources}/${v.n_cited} lanes=${v.lanes_total} subs=${(v.subclaims||[]).length}`);
  console.log('cost shown in footer :', await p.$eval('#vfoot', e => /cost/i.test(e.textContent)), '(want false)');

  await p.click('#cmpbtn');
  await p.waitForFunction(() => document.querySelectorAll('.wtab tr').length > 1, { timeout: 90000 });
  console.log('weight table rows    :', (await p.$$('.wtab tr')).length);
  console.log('headline             :', await p.$eval('.verdicthead', e => e.textContent.trim()));
  const rows = await p.$$eval('.wtab tr', trs => trs.slice(1).map(tr =>
    [...tr.querySelectorAll('td')].map(td => td.textContent.trim())));
  rows.forEach(r => console.log(`   ${r[0].padEnd(26)} | ${r[1].slice(0,42).padEnd(42)} | ${r[2].slice(0,52)}`));
  await p.screenshot({path:'/tmp/cc-compare.png', fullPage:true});
  await b.close();
})();
