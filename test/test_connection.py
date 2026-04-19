import docker
import json

# 测试连接到 NAS 的 Docker
NAS_IP = '192.168.3.2'
NAS_PORT = 2375

try:
    client = docker.DockerClient(
        base_url=f'tcp://{NAS_IP}:{NAS_PORT}',
        timeout=10
    )
    
    # Ping 测试
    client.ping()
    print(f"✓ 成功连接到 NAS Docker: {NAS_IP}:{NAS_PORT}")
    
    # 获取 Docker 版本信息
    info = client.version()
    print(f"\n📦 Docker 版本信息:")
    print(json.dumps(info, indent=2, default=str))
    
    # 列出所有运行中的容器
    containers = client.containers.list()
    print(f"\n🐳 运行中的容器数: {len(containers)}")
    for container in containers:
        print(f"  - {container.name}: {container.status}")
    
except Exception as e:
    print(f"✗ 连接失败: {e}")
    print(f"\n排查建议:")
    print(f"1. 确认 NAS 是否开启了 Docker TCP 监听")
    print(f"2. 执行: sudo netstat -tulnp | grep 2375")
    print(f"3. 检查防火墙设置")