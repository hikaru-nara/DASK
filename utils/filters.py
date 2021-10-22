import os
import re
class defaultGraphFilter():
	def __init__(self, args):
		self.filtermethod = "default"
		self.refilter = args.refilter
		self.triplets = []

	def filter(self, graphpaths):
		filtered_paths = [self.filtername(gp) for gp in graphpaths]
		for fp,gp in zip(filtered_paths,graphpaths):
			if not os.path.exists(fp) or self.refilter:
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
					if self.anomaly_detected(lst[:3]):
						continue
					g.write('\t'.join(lst[:3])+'\n')
					self.triplets.append(lst[:3])

	def anomaly_detected(self, triplet):
		if triplet in self.triplets:
			return True
		if triplet[0] in triplet[1] or \
			triplet[0] in triplet[2] or \
			triplet[1] in triplet[0] or \
			triplet[1] in triplet[2] or \
			triplet[2] in triplet[0] or \
			triplet[2] in triplet[1]:
			return True 
		if re.search(r'\W', triplet[0]) or \
			re.search(r'\W', triplet[1]) or \
			re.search(r'\W', triplet[2]):
			return True 
		return False

class confidenceFilter(defaultGraphFilter):
	def __init__(self, args):
		super().__init__(args)
		self.filtermethod = "conf"
		self.conf_thres = args.filter_conf

	def filtername(self, gp):
		return gp + "_conf{}".format(self.conf_thres) 

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
					if self.anomaly_detected(lst[:3]):
						continue
					if conf>self.conf_thres: 
						self.triplets.append(lst[:3])
						g.write('\t'.join(lst[:3])+'\n')

class confidenceFilter_uncased(defaultGraphFilter):
	def __init__(self, args):
		super().__init__(args)
		self.filtermethod = "conf_uncased"
		self.conf_thres = args.filter_conf

	def filtername(self, gp):
		return gp + "_conf{}_uncased".format(self.conf_thres) 

	def filter_(self, gp, fp):
		with open(gp, 'r', encoding='utf-8') as f:
			with open(fp, 'w') as g:
				for line in f:
					line = line.lower()
					lst = line.split('\t')
					if(len(lst)<=3):
						continue
					try:
						conf = float(lst[3][:-1])
					except:
						continue
					if self.anomaly_detected(lst[:3]):
						continue
					if conf>self.conf_thres: 
						self.triplets.append(lst[:3])
						g.write('\t'.join(lst[:3])+'\n')

filter_factory = {
	"default":defaultGraphFilter,
	"conf":confidenceFilter,
	'conf_uncased': confidenceFilter_uncased
}

if __name__ == "__main__":
	import numpy as np 
	conf_lst = []
	gp = 'data/results/db_org'
	with open(gp,'r') as f:
		for line in f:
			lst = line.split('\t')
			if(len(lst)<=3):
				continue
			try:
				conf = float(lst[3][:-1])
			except:
				continue
			conf_lst.append(conf)
	print(np.histogram(np.array(conf_lst),10))
