/* Deterministic local Canvas → JPEG pipe → FFmpeg renderer. No hosted services. */
const fs=require('node:fs');
const path=require('node:path');
const {pathToFileURL}=require('node:url');
const {spawn}=require('node:child_process');
const {once}=require('node:events');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const dir=__dirname;
const output=process.env.VIDEO_OUTPUT||path.join(dir,'design61-intro.mp4');
const ffmpeg=process.env.FFMPEG_PATH||'ffmpeg';
const fps=24,duration=36;
(async()=>{
  const launch={headless:true,args:['--disable-gpu']};
  if(process.env.CHROME_PATH)launch.executablePath=process.env.CHROME_PATH;
  const browser=await chromium.launch(launch);
  let encoder;
  try{
    const page=await browser.newPage({viewport:{width:1920,height:1200},deviceScaleFactor:1});
    await page.goto(pathToFileURL(path.join(dir,'index.html')).href+'?render=1');
    await page.evaluate(()=>document.fonts.ready);
    let errors=[];
    page.on('pageerror',e=>errors.push(e.message));
    const canvas=page.locator('#film');
    for(const [name,t] of [['poster.png',2.5],['sampling.png',8.5],['sequence.png',14.5],['media.png',20.5],['folder.png',26.5],['closing.png',32.5]]){
      await page.evaluate(t=>window.renderAt(t),t);
      await canvas.screenshot({path:path.join(dir,name)});
    }
    const args=['-hide_banner','-loglevel','warning','-y','-f','image2pipe','-vcodec','mjpeg','-r',String(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','medium','-crf','18','-pix_fmt','yuv420p','-threads','4','-movflags','+faststart',output];
    encoder=spawn(ffmpeg,args,{stdio:['pipe','ignore','inherit'],windowsHide:true});
    let encoderError;encoder.on('error',e=>encoderError=e);encoder.stdin.on('error',e=>encoderError=e);
    const complete=new Promise((resolve,reject)=>{encoder.once('close',code=>code===0?resolve():reject(new Error('FFmpeg exit '+code)));encoder.once('error',reject)});
    for(let frame=0;frame<fps*duration;frame++){
      if(encoderError)throw encoderError;
      const data=await page.evaluate(t=>{window.renderAt(t);return document.getElementById('film').toDataURL('image/jpeg',.94).split(',')[1]},frame/fps);
      if(!encoder.stdin.write(Buffer.from(data,'base64')))await once(encoder.stdin,'drain');
      if(frame%(fps*3)===0)console.log('Rendered '+Math.floor(frame/fps)+' / '+duration+' seconds');
    }
    encoder.stdin.end();await complete;
    if(errors.length)throw new Error(errors.join('\n'));
    console.log('MP4 complete:',output,fs.statSync(output).size+' bytes');
  }catch(e){if(encoder)encoder.kill();process.exitCode=1;console.error(e.message)}finally{await browser.close()}
})();
