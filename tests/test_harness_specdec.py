import importlib.util,json,struct,sys
from pathlib import Path
import pytest
from dental_agent.harness.checkpoints import tensor_headers,tensor_hash,audit_mtp,preserve_mtp
from dental_agent.harness.specdec import speculative_config,server_command,prometheus,delta,episode_measurements,equivalence,encode_messages

def safefile(path,tensors):
    path.parent.mkdir(parents=True,exist_ok=True);header={};blob=b''
    for name,value in tensors.items():
        header[name]=dict(dtype='U8',shape=[len(value)],data_offsets=[len(blob),len(blob)+len(value)]);blob+=value
    raw=json.dumps(header).encode();raw+=b' '*((-len(raw))%8)
    path.write_bytes(struct.pack('<Q',len(raw))+raw+blob)

def test_mtp_preservation_and_corruption(tmp_path):
    base=tmp_path/'base';merged=tmp_path/'merged'
    safefile(base/'model.safetensors',{'model.weight':b'abc','mtp.fc.weight':b'xyz'})
    safefile(merged/'model.safetensors',{'model.weight':b'def'})
    assert not audit_mtp(base,merged)['ok']
    assert preserve_mtp(base,merged)['ok']
    assert tensor_hash(tensor_headers(merged)['model.weight'])!=tensor_hash(tensor_headers(base)['model.weight'])
    assert preserve_mtp(base,merged)['ok']
    p=merged/'mtp-preserved.safetensors';s=p.read_bytes();p.write_bytes(s[:-1]+b'q')
    assert not audit_mtp(base,merged)['ok']
    with pytest.raises(ValueError):preserve_mtp(base,merged)

def test_bad_headers(tmp_path):
    (tmp_path/'model.safetensors').write_bytes(b'bad')
    with pytest.raises(ValueError):tensor_headers(tmp_path)

def test_conditions_same_common_arguments():
    a=server_command('vllm','/model','baseline',9000)
    b=server_command('vllm','/model','mtp_3',9000)
    assert b[:len(a)]==a
    assert speculative_config('ngram_2')==dict(method='ngram',num_speculative_tokens=2,prompt_lookup_min=1,prompt_lookup_max=2)
    with pytest.raises(ValueError):speculative_config('lossy')

def test_missing_metrics_and_counter_delta():
    a=prometheus('vllm:request_prefill_time_seconds_sum{model="x"} 1.2\n')
    b=prometheus('vllm:request_prefill_time_seconds_sum{model="x"} 2.5\n')
    assert delta(a,b,'vllm:request_prefill_time_seconds_sum')==pytest.approx(1.3)
    assert delta({},b,'vllm:request_prefill_time_seconds_sum') is None
    r=episode_measurements([{'generated_tokens':2,'wall_seconds':1}],2,.1)
    assert r['prefill_seconds'] is None and r['acceptance_rate'] is None and r['tokens_per_second']==1

def test_equivalence_does_not_hide_divergence():
    a=[dict(image_id=99999,final_answer=[],format_ok=True,turns=[dict(raw_output='a',tool_calls=[])])]
    assert equivalence(a,a)['passed']
    b=json.loads(json.dumps(a));b[0]['turns'][0]['raw_output']='b'
    assert not equivalence(a,b)['passed']
    assert not equivalence(a,[])['passed']

def test_image_transport_is_lossless():
    from PIL import Image
    import base64,io
    image=Image.new('RGB',(31,17),(10,20,30))
    message=encode_messages([dict(role='user',content=[dict(type='image',image=image),dict(type='text',text='same')])])[0]
    restored=Image.open(io.BytesIO(base64.b64decode(message['content'][0]['image_url']['url'].split(',')[1])))
    assert restored.size==image.size and restored.tobytes()==image.tobytes()

def test_table_does_not_report_failed_speedup():
    path=Path(__file__).parents[1]/'scripts/paper/generate_serving_table.py'
    spec=importlib.util.spec_from_file_location('serving_table',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    assert all(x['speedup'] is None for x in m.rows({}))
    report=dict(conditions=dict(baseline=dict(status='complete',episodes=[dict(measurements=dict(wall_seconds=2,generated_tokens=10,prefill_seconds=1,decode_seconds=1))]),mtp_1=dict(status='complete',episodes=[dict(measurements=dict(wall_seconds=1))])),gates=dict(mtp_1=dict(passed=False)))
    assert next(x for x in m.rows(report) if x['condition']=='mtp_1')['speedup'] is None

def test_serving_http_client_counters(tmp_path):
    from http.server import BaseHTTPRequestHandler,HTTPServer
    import threading
    from dental_agent.harness.specdec import ServingReply
    received=[]
    class Handler(BaseHTTPRequestHandler):
        counter=0
        def log_message(self,*args):pass
        def do_GET(self):
            body=f'vllm:request_prefill_time_seconds_sum {Handler.counter}\nvllm:request_decode_time_seconds_sum {Handler.counter*2}\n'.encode()
            self.send_response(200);self.end_headers();self.wfile.write(body)
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            Handler.counter+=1
            data=dict(choices=[dict(message=dict(content='{"final_answer":[]}'),finish_reason='stop')],usage=dict(prompt_tokens=3,completion_tokens=2),metrics=dict(speculative_decoding=dict(num_draft_tokens=4,num_accepted_draft_tokens=2)))
            self.send_response(200);self.end_headers();self.wfile.write(json.dumps(data).encode())
    server=HTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        reply=ServingReply(f'http://127.0.0.1:{server.server_port}')
        text,prompt,ids,pkv,cache=reply(None,None,[dict(role='system',content='same')],max_new_tokens=2048)
        assert text=='{"final_answer":[]}' and ids==[]
        assert received[0]['max_tokens']==2048 and received[0]['temperature']==0
        assert reply.requests[0]['prefill_seconds']==1 and reply.requests[0]['decode_seconds']==2
        assert reply.requests[0]['acceptance_rate']==.5
    finally:server.shutdown();server.server_close();thread.join(timeout=2)

@pytest.mark.parametrize('script',['check_mtp.py','merge_adapter.py','regenerate_train.py','mtp_finetune.py','probe_environment.py','spec_decode_bench.py','compare_acceptance.py'])
def test_cli_help_no_model_or_network(script):
    import subprocess
    root=Path(__file__).parents[1]
    done=subprocess.run([sys.executable,str(root/'scripts/bench'/script),'--help'],capture_output=True,text=True,timeout=30)
    assert done.returncode==0,done.stderr

def test_all_requested_serving_profiles_and_notebooks():
    import ast
    root=Path(__file__).parents[1]
    for name in ['VLM_Dental_Colab_Rules_Skills.ipynb','VLM_Dental_Colab_Speculative_Decoding.ipynb']:
        notebook=json.loads((root/'notebooks'/name).read_text())
        for i,cell in enumerate(notebook['cells']):
            if cell['cell_type']=='code':compile(''.join(cell['source']),f'{name}:{i}','exec')
        config=next(''.join(c['source']) for c in notebook['cells'] if 'PROFILES = {' in ''.join(c['source']))
        node=next(n for n in ast.parse(config).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PROFILES' for t in n.targets))
        profiles=eval(compile(ast.Expression(node.value),'profiles','eval'),{'dict':dict})
        assert profiles['dual_t4']['compute']=='fp16'
        assert profiles['rtx4090_qlora']['compute']=='bf16'
        assert profiles['rtxpro6000_48gb']['precision']=='bf16'
        assert profiles['rtxpro6000_48gb']['tag']==profiles['workstation48_lora']['tag']=='workstation48_lora_bf16'
    command=server_command('vllm','checkpoint','mtp_1',9000,tensor_parallel=2,dtype='float16')
    assert command[command.index('--dtype')+1]=='float16'
    assert command[command.index('--tensor-parallel-size')+1]=='2'
