# PROJDIR := $(realpath $(CURDIR)/..)
# DATADIR := $(PROJDIR)/data
# RAWDATADIR := $(DATADIR)/raw
# SRCDIR := $(PROJDIR)/src

BIINTERDATADIR := data/interim/binary_independent
BIPROCDATADIR := data/processed/binary_independent
BIINPUTS := $(BIPROCDATADIR)/X.pkl\
			$(BIPROCDATADIR)/y.pkl\
			$(BIPROCDATADIR)/X_train.pkl\
			$(BIPROCDATADIR)/X_test.pkl\
			$(BIPROCDATADIR)/y_train.pkl\
			$(BIPROCDATADIR)/y_test.pkl

.PHONY : dataset dataframe features inputs

all: inputs

dataset : data/raw/Occupancy_Estimation.csv	

dataframe : data/interim/raw_df.pkl

features : $(BIINTERDATADIR)/df.pkl

inputs: $(BIINPUTS)

data/raw/Occupancy_Estimation.csv :
	python src/data/download_dataset.py

data/interim/raw_df.pkl : data/raw/Occupancy_Estimation.csv
	python src/data/save_dataframe.py

$(BIINTERDATADIR)/df.pkl : data/interim/raw_df.pkl src/features/build_features.py
	python src/features/build_features.py -b -i

$(BIINPUTS) : $(BIINTERDATADIR)/df.pkl src/features/build_final_datasets.py
	python src/features/build_final_datasets.py -b -i