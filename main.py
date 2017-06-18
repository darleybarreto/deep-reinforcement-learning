from models import select_models

kwargs = {
			"load_path": False,\
			"display": False,\
			"episodes":10000,\
			"steps": 1000,\
			"n_processes": 1
		}

select_models.select_models(**kwargs)