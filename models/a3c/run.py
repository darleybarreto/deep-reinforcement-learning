import os
import sys
import torch.multiprocessing as mp
from .train import train
from .test import test

sys.path.insert(0,'../..')
from utils import menu

def main(model, opt, **kwargs):
	save_a3c = model[0] + "_a3c_model.pickle"
	save_txt_path = model[0] + "_a3c_scores.txt"

	save_a3c = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_files', save_a3c)
	save_txt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_files', save_txt_path)
	load_a3c = None
	# save_txt_path = None
	shared_model = A3C()
	shared_model.share_memory()
	
	optimizer = SharedAdam(shared_model.parameters(), lr=args.lr)
	optimizer.share_memory()
	processes = []
	num_processes = kwargs.get("n_processes", 10)

	p = mp.Process(target=test, args=(shared_model,\
					save_a3c, load_a3c, kwargs))
	p.start()
	processes.append(p)

	for rank in range(0, num_processes):
		p = mp.Process(target=train, args=(rank, shared_model, optimizer,\
						save_a3c, load_a3c, save_txt_path, kwargs))
	p.start()
	processes.append(p)

	for p in processes:
		p.join()

	# os.system("shutdown now -h")

if __name__ == '__main__':
	model, opt = menu()
	main(model, opt)