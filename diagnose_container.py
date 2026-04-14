import docker
import json

client = docker.DockerClient(base_url='tcp://192.168.3.2:2375', timeout=10)

# 获取 qBittorrent 容器
containers = client.containers.list(all=True)  # all=True 包括已停止的容器

for container in containers:
    print(f"\n{'='*60}")
    print(f"容器名: {container.name}")
    print(f"{'='*60}")
    print(f"状态: {container.status}")
    print(f"ID: {container.id[:12]}")
    print(f"镜像: {container.image.tags if hasattr(container.image, 'tags') else 'unknown'}")
    
    # 获取容器日志（最后 50 行）
    try:
        logs = container.logs(tail=50, stderr=True, stdout=True).decode('utf-8', errors='ignore')
        print(f"\n最近日志:")
        print(logs[-1000:] if len(logs) > 1000 else logs)  # 只显示最后 1000 字符
    except Exception as e:
        print(f"无法获取日志: {e}")
    
    # 获取容器的环境变量
    try:
        inspect_data = client.api.inspect_container(container.id)
        env_vars = inspect_data['Config']['Env']
        print(f"\n环境变量:")
        for env in env_vars[:5]:  # 只显示前 5 个
            print(f"  {env}")
    except Exception as e:
        print(f"无法获取环境变量: {e}")
    
    # 获取容器的端口映射
    try:
        inspect_data = client.api.inspect_container(container.id)
        ports = inspect_data['NetworkSettings']['Ports']
        print(f"\n端口映射:")
        for port, bindings in ports.items():
            print(f"  {port}: {bindings}")
    except Exception as e:
        print(f"无法获取端口信息: {e}")