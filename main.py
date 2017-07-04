from models import select_models

kwargs = {
			"load_path": True,\
			"display": False,\
			"episodes":5000,\
			"steps": 2000,\
			"n_processes": 3
		}

select_models.select_models(**kwargs)
