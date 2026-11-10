#!/usr/bin/env python3
"""Read-only local serving capability inventory: never loads weights or launches a server."""
import argparse,importlib.metadata,json,platform,re,shutil,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def nvidia_driver_info():
    """Read the host driver and advertised CUDA level without initializing CUDA."""
    try:
        done = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if done.returncode:
            return {'available': False, 'detail': (done.stderr or done.stdout).strip()[:500]}
        header = done.stdout + '\n' + done.stderr
        driver = re.search(r'Driver Version:\s*([0-9.]+)', header)
        cuda = re.search(r'CUDA Version:\s*([0-9.]+)', header)
        return {'available': True, 'driver_version': driver.group(1) if driver else None,
                'driver_cuda_api_max': cuda.group(1) if cuda else None}
    except Exception as exc:  # noqa: BLE001
        return {'available': False, 'detail': f'{type(exc).__name__}: {exc}'}


def serving_torch_cuda(vllm_executable):
    """Inspect the isolated serving environment's torch CUDA build if its interpreter is adjacent."""
    executable = Path(vllm_executable).expanduser()
    if not executable.is_absolute():
        found = shutil.which(str(vllm_executable))
        if not found:
            return None
        executable = Path(found)
    python = executable.resolve().parent / ('python.exe' if sys.platform == 'win32' else 'python')
    if not python.is_file():
        return None
    try:
        done = subprocess.run([str(python), '-c', 'import torch; print(torch.version.cuda or "cpu")'],
                              capture_output=True, text=True, timeout=30)
        return done.stdout.strip() if done.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def version_tuple(value):
    try:
        return tuple(int(part) for part in value.split('.')[:2])
    except (AttributeError, ValueError):
        return None


def cuda_compatibility_warning(serving_cuda, driver_info):
    """Explain CUDA runtime / host-driver compatibility without guessing GPU eligibility."""
    runtime = version_tuple(serving_cuda)
    driver = version_tuple(driver_info.get('driver_version'))
    host_cuda = driver_info.get('driver_cuda_api_max')
    host_cuda_version = version_tuple(host_cuda)
    if not runtime or not driver:
        return None
    if runtime[0] >= 13 and driver[0] < 580:
        return (f'Serving Torch uses CUDA {serving_cuda}, but host driver {driver_info.get("driver_version")} '
                'is below R580, NVIDIA’s CUDA 13.x minor-compatibility minimum. A CUDA 13 forward-compatibility '
                'package is not assumed: NVIDIA restricts forward compatibility to supported device classes, and '
                'this probe does not establish Quadro RTX 8000 eligibility.')
    if runtime[0] == 12 and driver[0] < 525:
        return (f'Serving Torch uses CUDA {serving_cuda}, but host driver {driver_info.get("driver_version")} '
                'is below R525, NVIDIA’s CUDA 12.x minor-compatibility minimum.')
    if host_cuda_version and runtime > host_cuda_version:
        if runtime[0] == 12:
            detail = 'CUDA 12.x minor-version compatibility may permit execution with a limited feature set; newer PTX/toolchain features can still fail.'
        else:
            detail = 'The host driver does not advertise this CUDA runtime level; validate driver compatibility before serving.'
        return (f'Serving Torch was built for CUDA {serving_cuda}, while nvidia-smi advertises CUDA {host_cuda}. '
                f'{detail} This probe does not certify model runtime.')
    return None


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--vllm-executable',default='vllm');p.add_argument('--output',required=True);p.add_argument('--gpu-indices',default='0');a=p.parse_args()
    from dental_agent.harness.evaluation import atomic_json
    from dental_agent.harness.specdec import gpu_snapshot
    packages={}
    for name in ['vllm','sglang','speculators','torch','transformers','peft']:
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    driver = nvidia_driver_info()
    serving_cuda = serving_torch_cuda(a.vllm_executable)
    report=dict(python=sys.version,platform=platform.platform(),packages=packages,gpus=gpu_snapshot(a.gpu_indices),
                nvidia_driver=driver,serving_torch_cuda_build=serving_cuda,
                available_vllm=shutil.which(a.vllm_executable),supported_cli_flags={},model_runtime_verified=False)
    warning = cuda_compatibility_warning(serving_cuda, driver)
    if warning:
        report['cuda_compatibility_warning'] = warning
    if report['available_vllm']:
        for label,cmd in [('version',[a.vllm_executable,'--version']),('help',[a.vllm_executable,'serve','--help'])]:
            done=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
            report[label]=dict(returncode=done.returncode,stdout=done.stdout,stderr=done.stderr)
        text=report['help']['stdout']+report['help']['stderr']
        report['supported_cli_flags']={flag:flag in text for flag in ['--speculative-config','--enable-per-request-metrics','--per-request-spec-decode-metrics']}
    # Analytical lower bound only, not a 4090 fit guarantee.
    report['reference_memory_lower_bound']=dict(checkpoint_bytes=19306216416,checkpoint_gib=19306216416/2**30,
        full_attention_kv_gib_at_16384=8*2*4*256*16384*2/2**30,
        excludes=['GDN state','vision/image activations','prefill workspace','CUDA graphs','MTP activations','allocator overhead'],
        reference='https://huggingface.co/Qwen/Qwen3.5-9B/raw/main/model.safetensors.index.json')
    atomic_json(a.output,report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
