// End-to-end: real check → Track this daily → real re-check → second history point.
const puppeteer = require('puppeteer-core');
(async () => {
  const b = await puppeteer.launch({
    executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    headless: 'new', args: ['--no-sandbox'] });
  const p = await b.newPage();
  p.on('pageerror', e => console.log('PAGE ERROR:', e.message));
  await p.goto('http://localhost:8899', { waitUntil: 'networkidle0' });
  await p.evaluate(() => localStorage.setItem('rcmnd_sid', 's_tracktest'));
  await p.reload({ waitUntil: 'networkidle0' });

  const CLAIM = 'Croatia adopted the euro in 2023';
  await p.type('#claim', CLAIM);
  await p.$eval('#deep', el => { el.checked = false; });   // fast lanes only
  const t0 = Date.now();
  await p.click('#go');
  await p.waitForFunction(() => lastVerdict && !lastVerdict.preliminary, { timeout: 120000 });
  const v = await p.evaluate(() => lastVerdict);
  console.log(`check done in ${((Date.now()-t0)/1000).toFixed(1)}s → ${v.label} score=${v.score} conf=${v.confidence}`);

  await p.click('#trackbtn');
  await p.waitForFunction(() => document.querySelectorAll('.trow').length === 1, { timeout: 30000 });
  console.log('after Track      :', await p.$eval('.trow .tsc', e => e.textContent),
              '| delta', await p.$eval('.trow .tdelta', e => e.textContent),
              '| bars', (await p.$$('.spark i')).length);

  console.log('re-checking live (runs a full verification again)…');
  const t1 = Date.now();
  await p.click('.trow [data-recheck]');
  await p.waitForFunction(() => document.querySelectorAll('.spark i').length === 2, { timeout: 180000 });
  console.log(`re-check done in ${((Date.now()-t1)/1000).toFixed(1)}s`);
  console.log('after Re-check   :', await p.$eval('.trow .tsc', e => e.textContent),
              '| delta', await p.$eval('.trow .tdelta', e => e.textContent),
              '| bars', (await p.$$('.spark i')).length);
  console.log('label line       :', await p.$eval('.trow .tl', e => e.textContent));

  // does it survive a reload? (session restore)
  await p.reload({ waitUntil: 'networkidle0' });
  await new Promise(r => setTimeout(r, 800));
  console.log('after reload     :', (await p.$$('.trow')).length, 'row(s) restored');
  await p.screenshot({path:'/tmp/cc-track.png', fullPage:true});
  await b.close();
})();
