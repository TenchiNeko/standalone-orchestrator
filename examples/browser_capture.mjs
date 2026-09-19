const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE);
const browser = await chromium.launch({headless:true});
const page = await browser.newPage({viewport:{width:1280,height:800}});
await page.goto(process.env.URL, {waitUntil:'networkidle'});
await page.screenshot({path:process.env.OUT});
await browser.close();
