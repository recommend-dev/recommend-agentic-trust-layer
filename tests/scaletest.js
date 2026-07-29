// The scale must start level, lean progressively as lanes land, and settle on the score.
const puppeteer = require('puppeteer-core');
(async () => {
  const b = await puppeteer.launch({executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    headless:'new', args:['--no-sandbox']});
  const p = await b.newPage();
  p.on('pageerror', e => console.log('PAGE ERROR:', e.message));
  await p.goto('http://localhost:8899', {waitUntil:'networkidle0'});

  const tilt = () => p.$eval('#beam', e => parseFloat((e.style.transform.match(/-?[\d.]+/) || [0])[0]));
  await p.type('#claim', 'Humans only use 10 percent of their brains');
  await p.$eval('#deep', el => { el.checked = false; });
  await p.click('#go');

  await p.waitForFunction(() => !document.getElementById('s-verdict').classList.contains('hidden'), {timeout:60000});
  console.log('scale visible immediately :', true);
  console.log('reasoning hidden by default:', await p.$eval('#reasoning', e => e.classList.contains('hidden')), '(want true)');
  console.log('starts level              :', await tilt(), '(want 0)');

  const seen = [];
  for (let i = 0; i < 60; i++) {
    const t = await tilt();
    if (!seen.length || Math.abs(seen[seen.length-1] - t) > 0.01) seen.push(t);
    if (await p.evaluate(() => lastVerdict && !lastVerdict.preliminary)) break;
    await new Promise(r => setTimeout(r, 500));
  }
  await new Promise(r => setTimeout(r, 1200));
  const v = await p.evaluate(() => lastVerdict);
  console.log('tilt steps (progressive)  :', seen.map(x => x.toFixed(1)).join(' → '));
  console.log('final tilt                :', await tilt(), '| score', v.score);
  console.log('dishes counter-rotate     :', await p.$eval('#dishL', e => e.style.transform));
  console.log('verdict                   :', v.label, v.score + '/100', 'conf', v.confidence);

  await p.click('#reasonbtn');
  await new Promise(r => setTimeout(r, 400));
  console.log('reasoning after button    :', !(await p.$eval('#reasoning', e => e.classList.contains('hidden'))), '(want true)');
  await p.$eval('#s-verdict', el => el.scrollIntoView());
  await new Promise(r => setTimeout(r, 500));
  await (await p.$('#s-verdict')).screenshot({path:'/tmp/cc-scale.png'});
  await b.close();
})();
