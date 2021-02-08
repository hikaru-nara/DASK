import os

class defaultGraphFilter():
	def __init__(self, args):
		self.filtermethod = "default"

	def filter(self, graphpaths):
		filtered_paths = [self.filtername(gp) for gp in graphpaths]
		for fp,gp in zip(filtered_paths,graphpaths):
			if not os.path.exists(fp):
				self.filter_(gp, fp)
		return filtered_paths

	def filtername(self, gp):
		return gp + "_" + self.filtermethod

	def filter_(self, gp, fp):
		with open(gp, 'r', encoding='utf-8') as f:
			with open(fp, 'w') as g:
				for line in f:
					lst = line.split('\t')
					# head, rel, tail, conf = lst[0], lst[1],lst[2],lst[3]
					g.write('\t'.join(lst[:3])+'\n')

class confidenceFilter(defaultGraphFilter):
	def __init__(self, args):
		self.filtermethod = "conf"
		self.conf_thres = args.filter_conf

	def filter_(self, gp, fp):
		with open(gp, 'r', encoding='utf-8') as f:
			with open(fp, 'w') as g:
				for line in f:
					lst = line.split('\t')
					if(len(lst)<=3):
						continue
					try:
						conf = float(lst[3][:-1])
					except:
						continue
					if conf>self.conf_thres: 
						g.write('\t'.join(lst[:3])+'\n')

filter_factory = {
	"default":defaultGraphFilter,
	"conf":confidenceFilter
}

if __name__ == "__main__":
	import numpy as np 
	conf_lst = []
	gp = 'data/results/da_org'
	with open(gp,'r') as f:
		for line in f:
			lst = line.split('\t')
			if(len(lst)<=3):
				continue
			conf = float(lst[3][:-1])
			conf_lst.append(conf)
	print(np.histogram(np.array(conf_lst),10))
