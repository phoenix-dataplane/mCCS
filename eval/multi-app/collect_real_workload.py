import subprocess as sb
data = sb.getoutput('grep --recursive "Rank 0: run time" /tmp').split('\n')
# filter with /tmp/ at the beginning, and remove the first 5 characters
data = [i[5:] for i in data if i.find('/tmp/')==0]
txt = ['setting,job,jct']
for i in data:
    setting = '-'.join(i.split('/')[1].split('-')[2:])
    if setting.find('ecmp-qosv2')==-1:
        app = i.split('[')[1].split(']')[0]
        time = i.split(': ')[-1].split(' ')[0]
        txt.append(f'{setting},{app},{time}')
with open('../plot/data/real_workload.csv', 'w') as f:
    f.write('\n'.join(txt))
    