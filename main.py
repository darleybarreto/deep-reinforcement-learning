from models import select_models

kwargs = {
			"load_path": False,\
			"display": True,\
			"episodes":5000,\
			"steps": 2000,\
			"n_processes": 1
		}

select_models.select_models(**kwargs)