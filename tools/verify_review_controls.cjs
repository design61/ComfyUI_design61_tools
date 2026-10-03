/* Read a saved workflow and test real flow wiring without submitting a prompt. */
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const crypto=require('node:crypto');
const began=performance.now();
const root=path.resolve(__dirname,'..');
const workflow=process.argv[2]||path.join(root,'examples','MiniMax_H3_single_sampler_design61.json');
const original=fs.readFileSync(workflow);
const saved=JSON.parse(original);
const types={start:'H3ContinuumExternalSequenceStart_design61',end:'H3ContinuumExternalSequenceEnd_design61'};
const nodes=new Map();
const graph={links:Object.fromEntries(saved.links.map(([id,origin_id,origin_slot,target_id,target_slot,type])=>[id,{id,origin_id,origin_slot,target_id,target_slot,type}])),getNodeById(id){return nodes.get(id)}};
class Node {
  constructor(record){Object.assign(this,{id:record.id,type:record.type,comfyClass:record.type,inputs:record.inputs||[],outputs:record.outputs||[],widgets:[],graph});}
  getInputNode(slot){const link=graph.links[this.inputs[slot]?.link];return graph.getNodeById(link?.origin_id);}
  addWidget(type,name,value,callback,options){const widget={type,name,value,callback,options};this.widgets.push(widget);return widget;}
  computeSize(){return [360,420]} setSize(){} setDirtyCanvas(){}
}
for(const record of saved.nodes)nodes.set(record.id,new Node(record));
let extension;
const commands=[];
const sandbox={queueMicrotask,console,app:{registerExtension(value){extension=value},async queuePrompt(){const start=[...nodes.values()].find(n=>n.type===types.start);commands.push(Object.fromEntries(start.widgets.map(w=>[w.name,w.value])));}}};
vm.createContext(sandbox);
const source=fs.readFileSync(process.argv[3]||path.join(root,'web','external_sampling.js'),'utf8')
  .replace('import { app } from "../../scripts/app.js";','const app=globalThis.app;')
  .replace('import { refreshV39ReferenceImagesForSampler } from "./upstream_reference_images.js";','const refreshV39ReferenceImagesForSampler=()=>{};')
  .replaceAll('export function','function');
vm.runInContext(source,sandbox);
(async()=>{
  await extension.beforeRegisterNodeDef(Node,{name:types.end});
  const end=[...nodes.values()].find(n=>n.type===types.end);
  const start=typeof sandbox.findExternalSequenceStart==='function'?sandbox.findExternalSequenceStart(end):[...nodes.values()].find(n=>n.type===types.start);
  if(!start)throw Error('No Start resolved through saved flow wire');
  // Use only control widgets: no private prompt, filenames or model settings.
  start.widgets=[{name:'generation_mode',value:'Review Each Chunk'},{name:'review_action',value:'Start / Resume'},{name:'expected_revision',value:''}];
  end.onExecuted({external_review:[{mode:'Review Each Chunk',status:'review_ready',revision:'synthetic-r1',accepted:1,chunks:2,history:[{revision:'synthetic-r1',chunk:1,seed:0}]}]});
  const actions=['Use it and continue','Finish here','Try this chunk again','Start again from Chunk 1'];
  for(const action of actions){
    await end.widgets.find(w=>w.name===action).callback();
    const queued=commands.at(-1);
    if(queued?.review_action!==action||queued?.expected_revision!=='synthetic-r1')throw Error('Command was not queued: '+action);
  }
  const history=end.widgets.find(w=>w.name.startsWith('Render History'));
  history.callback();if(!end.widgets.some(w=>w.name==='Saved Takes'))throw Error('History did not open');
  history.callback();if(end.widgets.some(w=>w.name==='Saved Takes'))throw Error('History did not close');
  const route=[];let current=end;
  while(current!==start){route.push({id:current.id,type:current.type});const i=current.inputs.findIndex(s=>s.name==='flow');current=current.getInputNode(i>=0?i:0);}
  route.push({id:start.id,type:start.type});
  if(!fs.readFileSync(workflow).equals(original))throw Error('Workflow was modified');
  console.log(JSON.stringify({result:'PASS',workflow_sha256:crypto.createHash('sha256').update(original).digest('hex'),flow_route:route,actions,history_toggle:'PASS',actual_prompts_submitted:0,model_sampling:false,elapsed_seconds:Number(((performance.now()-began)/1000).toFixed(3))},null,2));
})().catch(error=>{console.error(error.message);process.exitCode=1});
