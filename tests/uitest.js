const puppeteer = require('puppeteer-core');
(async () => {
  const b = await puppeteer.launch({
    executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    headless: 'new', args: ['--no-sandbox'] });
  const p = await b.newPage();
  const errs = [];
  p.on('pageerror', e => errs.push(e.message));
  p.on('console', m => { if (m.type() === 'error') errs.push('console: ' + m.text()); });
  await p.goto('http://localhost:8899', { waitUntil: 'networkidle0' });

  // NB: offsetParent is null for position:fixed elements, so it can't be used here
  const vis = s => p.$eval(s, el => getComputedStyle(el).display !== 'none'
                                    && el.getBoundingClientRect().width > 0);
  console.log('modal visible on load :', await vis('#modal'), '(want false)');
  console.log('logo rendered         :', await p.$eval('.brand', el => el.getBoundingClientRect().width > 40));
  console.log('favicon link          :', await p.$eval('link[rel=icon]', el => el.href.slice(0,32) + '…'));
  console.log('mcp key shown         :', (await p.$eval('#mcp-key', el => el.textContent)).slice(0,20) + '…');
  console.log('mcp cli filled        :', (await p.$eval('#mcp-cli', el => el.textContent)).includes('claude mcp add'));
  console.log('stray bullet on sub   :', await p.$eval('.sheetsub', el =>
      getComputedStyle(el, ':before').content), '(want none)');

  // the compare button lives inside the verdict section, so it must not even be
  // reachable before a check has produced a verdict
  console.log('cmp button pre-check  :', await vis('#cmpbtn'), '(want false)');

  // simulate a finished check, then open + close the modal
  await p.evaluate(() => {
    document.getElementById('s-verdict').classList.remove('hidden');
    lastVerdict = {label:'REFUTED',confidence:0.84,p_true:0.04,
                   rationale:'test',n_sources:15}; lastClaim='test claim'; });
  await p.click('#cmpbtn');
  await new Promise(r => setTimeout(r, 400));
  console.log('modal after open      :', await vis('#modal'), '(want true)');
  await p.click('#mx');
  await new Promise(r => setTimeout(r, 300));
  console.log('modal after X         :', await vis('#modal'), '(want false)');
  await p.click('#cmpbtn'); await new Promise(r => setTimeout(r, 300));
  await p.keyboard.press('Escape'); await new Promise(r => setTimeout(r, 300));
  console.log('modal after Esc       :', await vis('#modal'), '(want false)');

  // MCP is now its own modal off the header, not part of the results flow
  console.log('mcp modal on load     :', await vis('#mcpmodal'), '(want false)');
  await p.click('#mcpopen'); await new Promise(r => setTimeout(r, 300));
  console.log('mcp modal after open  :', await vis('#mcpmodal'), '(want true)');
  await p.click('#mcpx'); await new Promise(r => setTimeout(r, 300));
  console.log('mcp modal after X     :', await vis('#mcpmodal'), '(want false)');
  console.log('mcp NOT in page flow  :', await p.$('#s-mcp') === null, '(want true)');

  // truth score + tracking
  await p.evaluate(() => {
    $('sfill').style.width = '0%';
    document.dispatchEvent(new Event('x'));
  });
  console.log('session id persisted  :', await p.evaluate(() => !!localStorage.getItem('rcmnd_sid')));
  const tracked = await p.evaluate(async () => {
    const sid = localStorage.getItem('rcmnd_sid');
    await fetch(`/api/track/add?sid=${sid}&claim=UI%20test%20claim&score=80&conf=.7&label=SUPPORTED`);
    await fetch(`/api/track/add?sid=${sid}&claim=UI%20test%20claim&score=64&conf=.6&label=LIKELY%20TRUE`);
    const d = await (await fetch(`/api/tracked?sid=${sid}`)).json();
    renderTracked(d.tracked);
    return d.tracked[0];
  });
  console.log('tracked delta         :', tracked.delta, '(want -16)');
  console.log('track section visible :', await vis('#s-track'), '(want true)');
  console.log('sparkline bars        :', await p.$$eval('.spark i', e => e.length), '(want 2)');
  await p.evaluate(async () => {
    const sid = localStorage.getItem('rcmnd_sid');
    await fetch(`/api/track/remove?sid=${sid}&claim=UI%20test%20claim`);
  });

  await p.click('#howopen'); await new Promise(r => setTimeout(r, 300));
  console.log('how modal opens       :', await vis('#howmodal'), '(want true)');
  console.log('how modal steps       :', await p.$$eval('#howmodal .step', e => e.length), '(want 7)');
  console.log('no vendor names in UI :', await p.evaluate(() =>
      !/exa|serpapi|parallel\.ai|apify|diffbot/i.test(document.body.innerText)), '(want true)');
  await p.click('#howx'); await new Promise(r => setTimeout(r, 300));
  console.log('how modal closes      :', await vis('#howmodal'), '(want false)');

  console.log('JS errors             :', errs.length ? errs : 'none');
  await p.setViewport({width:390,height:800});
  await p.screenshot({path:'/tmp/cc-mobile.png'});
  await p.setViewport({width:1280,height:1000});
  await p.screenshot({path:'/tmp/cc-desktop.png', fullPage:true});
  await b.close();
})();
