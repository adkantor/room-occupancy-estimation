
BININTERDATADIR := data/interim/binary
BINPROCDATADIR := data/processed/binary
BININPUTS := $(BINPROCDATADIR)/X.pkl\
			 $(BINPROCDATADIR)/y.pkl\
			 $(BINPROCDATADIR)/X_train.pkl\
			 $(BINPROCDATADIR)/X_test.pkl\
			 $(BINPROCDATADIR)/y_train.pkl\
			 $(BINPROCDATADIR)/y_test.pkl

MCINTERDATADIR := data/interim/multiclass
MCPROCDATADIR := data/processed/multiclass
MCINPUTS := $(MCPROCDATADIR)/X.pkl\
			$(MCPROCDATADIR)/y.pkl\
			$(MCPROCDATADIR)/X_train.pkl\
			$(MCPROCDATADIR)/X_test.pkl\
			$(MCPROCDATADIR)/y_train.pkl\
			$(MCPROCDATADIR)/y_test.pkl

MODELS := models/binary/svc-binary.sav\
		  models/binary/knn-binary.sav\
		  models/binary/cart-binary.sav\
		  models/binary/rf-binary.sav\
		  models/binary/adaboost-binary.sav

.PHONY : dataset dataframe features inputs models

all: inputs

dataset : data/raw/Occupancy_Estimation.csv	

dataframe : data/interim/raw_df.pkl

features : $(BININTERDATADIR)/df.pkl $(MCINTERDATADIR)/df.pkl

inputs: $(BININPUTS) $(MCINPUTS)

models: $(MODELS)

data/raw/Occupancy_Estimation.csv :
	python src/data/download_dataset.py

data/interim/raw_df.pkl : data/raw/Occupancy_Estimation.csv
	python src/data/save_dataframe.py

$(BININTERDATADIR)/df.pkl : data/interim/raw_df.pkl src/features/build_features.py
	python src/features/build_features.py -b

$(MCINTERDATADIR)/df.pkl : data/interim/raw_df.pkl src/features/build_features.py
	python src/features/build_features.py -m

$(BININPUTS) : $(BININTERDATADIR)/df.pkl src/features/build_final_datasets.py
	python src/features/build_final_datasets.py -b

$(MCINPUTS) : $(MCINTERDATADIR)/df.pkl src/features/build_final_datasets.py
	python src/features/build_final_datasets.py -m

models/binary/svc-binary.sav : $(BININPUTS) src/models/train_models.py
	python src/models/train_models.py -X $(BINPROCDATADIR)/X.pkl -y $(BINPROCDATADIR)/y.pkl -b --svc

models/binary/knn-binary.sav : $(BININPUTS) src/models/train_models.py
	python src/models/train_models.py -X $(BINPROCDATADIR)/X.pkl -y $(BINPROCDATADIR)/y.pkl -b --knn

models/binary/cart-binary.sav : $(BININPUTS) src/models/train_models.py
	python src/models/train_models.py -X $(BINPROCDATADIR)/X.pkl -y $(BINPROCDATADIR)/y.pkl -b --cart

models/binary/rf-binary.sav : $(BININPUTS) src/models/train_models.py
	python src/models/train_models.py -X $(BINPROCDATADIR)/X.pkl -y $(BINPROCDATADIR)/y.pkl -b --rf

models/binary/adaboost-binary.sav : $(BININPUTS) src/models/train_models.py
	python src/models/train_models.py -X $(BINPROCDATADIR)/X.pkl -y $(BINPROCDATADIR)/y.pkl -b --adaboost