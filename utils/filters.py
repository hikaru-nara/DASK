import os

class defaultGraphFilter():
	def __init__(self, args):
		self.filtermethod = "default";

	def filter(self, graphpaths):
		filtered_paths = [self.filtername(gp) for gp in graphpaths]
		for fp,gp in zip(filtered_paths,graphpaths):
			if not os.path.exists(fp):
				self.filter_(gp, fp)
		return filtered_paths

	def filtername(self, gp):
		return gp + "_" + "default"

	def filter_(self, gp, fp):
		with open(gp, 'r', encoding='utf-8') as f:
			with open(fp, 'w') as g:
				for line in f:
					lst = line.split('\t')
					# head, rel, tail, conf = lst[0], lst[1],lst[2],lst[3]
					g.write('\t'.join(lst[:3])+'\n')

filter_factory = {
	"default":defaultGraphFilter
}