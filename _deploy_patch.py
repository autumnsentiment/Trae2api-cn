import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.5.246', username='user', password='qb46475166.', timeout=20)

remote = '/vol2/1000/trae-cn-relay'
files = ['src/main.py', 'src/auth.py', 'src/trae_client.py', 'src/trae_decrypt.py', 'docker-compose.yml']

stdin, stdout, stderr = ssh.exec_command(f'mkdir -p {remote}/src', timeout=30)
stdout.read(); stderr.read()

sftp = ssh.open_sftp()
for rel in files:
    sftp.put(rel, f'{remote}/{rel}')
    print('uploaded', rel)
sftp.close()

# Rebuild container with sudo password via stdin
cmd = f'cd {remote} && sudo -S docker compose up -d --build trae-cn-relay'
stdin, stdout, stderr = ssh.exec_command(cmd, timeout=600)
stdin.write('qb46475166.\n')
stdin.flush()
out = stdout.read().decode('utf-8', 'replace')
err = stderr.read().decode('utf-8', 'replace')
print(out)
if err:
    print('STDERR:', err[-3000:])
code = stdout.channel.recv_exit_status()
print('exit', code)
ssh.close()
