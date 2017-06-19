from models import select_models

kwargs = {
			"load_path": False,\
			"display": False,\
			"episodes":10000,\
			"steps": "inf",\
			"n_processes": 1
		}

select_models.select_models(**kwargs)