import json
from pathlib import Path
import shutil
import subprocess

import pytest


def test_review_ui_uses_backend_status_and_queues_revision_bound_actions(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for frontend behavior verification")
    source = (Path(__file__).resolve().parents[1] / "web/external_sampling.js").read_text(encoding="utf-8")
    source = source.replace('import { app } from "../../scripts/app.js";', 'const app = globalThis.app;').replace("export function", "function")
    source = source.replace('import { refreshV39ReferenceImagesForSampler } from "./upstream_reference_images.js";', 'const refreshV39ReferenceImagesForSampler = () => {};')
    script = """
let extension, queued;
const start = {comfyClass: 'H3ContinuumExternalSequenceStart_design61', widgets: [{name:'review_action',value:'Start / Resume'},{name:'expected_revision',value:''}]};
globalThis.app = {registerExtension: value => extension = value, queuePrompt: async () => queued = start.widgets.map(item => item.value)};
""" + source + """
class End {
  constructor() {this.widgets=[];this.inputs=[{name:'flow',link:7}];}
  getInputNode() {return start;}
  addWidget(type,name,value,callback,options) {const widget={type,name,value,callback,options}; this.widgets.push(widget); return widget;}
  computeSize() {return [300,200];}
  setSize() {}
  setDirtyCanvas() {}
}
await extension.beforeRegisterNodeDef(End, {name:'H3ContinuumExternalSequenceEnd_design61'});
const end=new End();
end.onExecuted({external_review:[{status:'in_progress',revision:'r0',accepted:1,chunks:4,history:[]}]});
if(end.widgets.some(item=>item.type==='button')) throw Error('in-progress exposed review buttons');
end.onExecuted({external_review:[{status:'review_ready',revision:'r1',accepted:1,chunks:4,history:[{revision:'r1',chunk:1,seed:3}]}]});
const actions=end.widgets.filter(item=>item.type==='button'&&!item.name.startsWith('Render History'));
if(actions.length!==4) throw Error('review-ready actions missing');
await actions.find(item=>item.name==='Try this chunk again').callback();
if(JSON.stringify(queued)!==JSON.stringify(['Try this chunk again','r1'])) throw Error('action not bound to canonical revision');
if(end.widgets.some(item=>item.serialize!==false||!item.options.tooltip)) throw Error('presentation state serialized or help missing');
end.onExecuted({external_review:[{status:'complete',revision:'r2',accepted:4,chunks:4,review_unit:{chunk:4},history:[{revision:'r2',chunk:4,seed:4}]}]});
if(end.widgets.some(item=>['Use it and continue','Use it and finish the rest'].includes(item.name))) throw Error('completion has no-op actions');
if(!end.widgets.some(item=>item.name==='Try this chunk again')) throw Error('valid completion review unit lost retry');
end.onExecuted({external_review:[{status:'complete',revision:'r3',accepted:4,chunks:4,history:[]}]});
if(end.widgets.some(item=>item.type==='button')) throw Error('missing backend review unit invented retry/history');
end.onExecuted({external_review:[{mode:'Full Video',status:'complete',revision:'rf',accepted:4,chunks:4,review_unit:{chunk:4},history:[{revision:'rf',chunk:4,seed:4}]}]});
if(end.widgets.some(item=>item.type==='button')) throw Error('Full Video exposed Review or History buttons');
end.onExecuted({external_review:[{mode:'Review Each Chunk',status:'review_ready',revision:'rs',accepted:1,chunks:4,review_unit:{chunk:1},history:[]}]});
await end.widgets.find(item=>item.name==='Finish here').callback();
if(JSON.stringify(queued)!==JSON.stringify(['Finish here','rs'])) throw Error('Finish command changed to running remaining chunks');
await end.widgets.find(item=>item.name==='Start again from Chunk 1').callback();
if(queued[0]!=='Start again from Chunk 1') throw Error('Restart all command missing');
// Multiple legacy Reroute nodes must resolve the same controller. Callback
// bindings must also survive changing connections and replacing Start widgets.
const routed=new End();
const route1={id:101,type:'Reroute',inputs:[{name:'',link:8}],getInputNode(){return route2;}};
const route2={id:102,type:'Reroute',inputs:[{name:'',link:9}],getInputNode(){return start;}};
routed.getInputNode=()=>route1;
routed.onExecuted({external_review:[{mode:'Review Each Chunk',status:'review_ready',revision:'rerouted-revision',accepted:1,chunks:2,history:[{revision:'take-routed',chunk:1,seed:3}]}]});
for(const name of ['Use it and continue','Finish here','Try this chunk again','Start again from Chunk 1']) {
  queued=null;
  start.widgets=[{name:'review_action',value:'Start / Resume'},{name:'expected_revision',value:''}];
  await routed.widgets.find(item=>item.name===name).callback();
  if(JSON.stringify(queued)!==JSON.stringify([name,'rerouted-revision'])) throw Error('Rerouted '+name+' failed');
}
const historyButton=routed.widgets.find(item=>item.name.startsWith('Render History'));
historyButton.callback();
if(!routed.widgets.some(item=>item.name==='Saved Takes')) throw Error('History failed to open');
historyButton.callback();
if(routed.widgets.some(item=>item.name==='Saved Takes')) throw Error('History failed to close');
route2.getInputNode=()=>route1;
if(findExternalSequenceStart(routed)!==null) throw Error('Cyclic flow hung or chose unrelated Start');
queued=null;
await routed.widgets.find(item=>item.name==='Use it and continue').callback();
if(queued!==null||!routed.widgets.find(item=>item.name==='Review status').value.includes('Cannot find')) throw Error('Missing controller failed silently');
route2.getInputNode=()=>start;
const originalQueue=app.queuePrompt;
app.queuePrompt=async()=>{throw Error('synthetic queue failure');};
await routed.widgets.find(item=>item.name==='Use it and continue').callback();
if(!routed.widgets.find(item=>item.name==='Review status').value.includes('synthetic queue failure')) throw Error('Queue error failed silently');
app.queuePrompt=originalQueue;
class Start {constructor(){this.widgets=[{name:'chunks',value:4},{name:'chunk_seconds',value:5},{name:'generation_mode',value:'Review by Chunk'},{name:'review_action',value:'Start / Resume',type:'combo'},{name:'expected_revision',value:'',type:'text'}];}}
await extension.beforeRegisterNodeDef(Start,{name:'H3ContinuumExternalSequenceStart_design61'});
const setup=new Start();setup.onNodeCreated();
if(!setup.widgets.find(item=>item.name==='review_action').hidden) throw Error('Internal review command visible in Start');
if(setup._externalProgress.total_seconds!==20) throw Error('Total length not calculated live');
setup.widgets[1].value=7;setup.onDrawForeground();
if(setup._externalProgress.total_seconds!==28) throw Error('Duration failed to refresh after editing seconds');
const flowNode={id:10,comfyClass:'H3ContinuumExternalSequenceStart_design61',connect(out,node,slot){node.inputs[slot].link=123;}};
const prep={id:20,comfyClass:'H3ContinuumExternalPrepare_design61',inputs:[{name:'sequence_flow',link:99}]};
const cond={id:30,comfyClass:'H3ContinuumExternalConditioning_design61',inputs:[{name:'sequence_flow',link:null}],outputs:[{links:[98]}]};
cond.graph={links:{98:{target_id:20},99:{origin_id:10}},getNodeById(id){return id===10?flowNode:prep;}};
if(!bindExternalConditioningFlow(cond)||cond.inputs[0].link!==123) throw Error('Recreated Conditioning did not recover matching automatic Flow');
if(bindExternalConditioningFlow(cond)) throw Error('Explicit Flow overwritten');
globalThis.document={createElement(tag){return {tag,textContent:'',style:{},children:[],append(...items){this.children.push(...items);},replaceChildren(){this.children=[];}};}};
class PanelStart extends Start {
  constructor(){super();this.comfyClass='H3ContinuumExternalSequenceStart_design61';this.outputs=[{links:[500]}];}
  addDOMWidget(name,type,host,options){const item={name,type,options,host};this.widgets.push(item);return item;}
  computeSize(){return [340,500];} setSize(size){this.size=size;} setDirtyCanvas(){}
}
const panel=new PanelStart();panel.onNodeCreated();
if(!panel._externalStatusHost||panel.size[1]!==500) throw Error('Status panel has no reserved node height');
const domWidget=panel.widgets.find(item=>item.name==='sequence_status');
if(domWidget.options.serialize!==false||!domWidget.options.tooltip) throw Error('Status serialized or missing help');
panel.onExecuted({external_progress:[{chunks:4,completed:1,current:2,remaining:3,total_seconds:20,status:'running'}]});
const elements=[];function collect(element){elements.push(element.textContent);element.children.forEach(collect);}collect(panel._externalStatusHost);
if(!['TOTAL LENGTH','20 s','TOTAL','DONE','CURRENT','LEFT','Running chunk 2 / 4'].every(text=>elements.includes(text))) throw Error('Backend counters or length missing from DOM');
const panelEnd=new End();panelEnd.getInputNode=()=>panel;panelEnd.comfyClass='H3ContinuumExternalSequenceEnd_design61';
panel.graph={links:{500:{target_id:7}},getNodeById(){return panelEnd;}};
panelEnd.onExecuted({external_review:[{mode:'Review Each Chunk',status:'review_ready',revision:'rp',accepted:1,chunks:4,history:[],progress:{chunks:4,completed:1,current:1,remaining:3,total_seconds:20,status:'review_ready'}}]});
panel.widgets.find(item=>item.name==='generation_mode').value='Full Video';panel.onDrawForeground();
if(panelEnd.widgets.some(item=>item.type==='button')) throw Error('Switching to Full left Review buttons visible');
panel.widgets.find(item=>item.name==='chunk_seconds').value=7;panel.onDrawForeground();
if(panel._externalProgress.total_seconds!==28) throw Error('Old backend progress overwrote edited planned duration');
const modeReroute={type:'Reroute',inputs:[{name:'',link:500}],outputs:[{links:[501]}]};
modeReroute.graph=panel.graph;
modeReroute.getInputNode=()=>panel;
panelEnd.getInputNode=()=>modeReroute;
panel.graph={links:{500:{target_id:8},501:{target_id:7}},getNodeById(id){return id===8?modeReroute:panelEnd;}};
modeReroute.graph=panel.graph;
panel.widgets.find(item=>item.name==='generation_mode').value='Review Each Chunk';panel.onDrawForeground();
if(!panelEnd.widgets.some(item=>item.name==='Use it and continue')) throw Error('Rerouted Review mode buttons missing');
panel.widgets.find(item=>item.name==='generation_mode').value='Full Video';panel.onDrawForeground();
if(panelEnd.widgets.some(item=>item.type==='button')) throw Error('Rerouted Full mode left Review/History visible');
process.stdout.write(JSON.stringify({result:'PASS'}));
"""
    path = tmp_path / "external-review.mjs"
    path.write_text(script, encoding="utf-8")
    result = subprocess.run([node, str(path)], capture_output=True, text=True, encoding="utf-8", timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["result"] == "PASS"
