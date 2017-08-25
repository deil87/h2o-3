from builtins import range
import sys, os
sys.path.insert(1, "../../../")
import h2o
from tests import pyunit_utils
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
import random

from enum import Enum

from h2o.estimators.gbm import H2OGradientBoostingEstimator

# make GBM model
#   h2o_df = h2o.import_file(path=pyunit_utils.locate("smalldata/logreg/prostate_train.csv"))
#   h2o_df["CAPSULE"] = h2o_df["CAPSULE"].asfactor()
#   model=H2OGradientBoostingEstimator(distribution="bernoulli",
#                                    ntrees=100,
#                                    max_depth=4,
#                                    learn_rate=0.1)
#   model.train(y="CAPSULE",
#             x=["AGE","RACE","PSA","GLEASON"],
#             training_frame=h2o_df)
#
#   pathToSave = os.getcwd()
#   h2o.save_model(model, path=pathToSave, force=True)  # save model in order to compare mojo and h2o predict output
#   modelfile = model.download_mojo(path=pathToSave, get_genmodel_jar=True)
#   print("Model saved to "+modelfile)
# These variables can be tweaked to increase / reduce stress on the test. However when submitting to GitHub
# please keep these reasonably low, so that the test wouldn't take exorbitant amounts of time.


def deeplearning_mojo():

    # h2o_data = h2o.upload_file(path=pyunit_utils.locate("smalldata/logreg/prostate.csv"))
    # h2o_data.summary()
    # parmsGLM = {'family':'binomial', 'alpha':0.5, 'standardize':True}
    # pyunit_utils.javapredict("glm", "class", h2o_data, h2o_data, list(range(2, h2o_data.ncol)), 1, pojo_model=False,
    #                          **parmsGLM)

    allAct = Enum("maxout", "rectifier", "maxoutwithdropout", "tanhwithdropout", "rectifierwithdropout", "tanh")
    prostate = h2o.import_file(path=pyunit_utils.locate("smalldata/logreg/prostate_missing.csv"))
    prostate[1] = prostate[1].asfactor()  # CAPSULE -> CAPSULE
    prostate[2] = prostate[2].asfactor()  # AGE -> Factor
    prostate[3] = prostate[3].asfactor()  # RACE -> Factor
    prostate[4] = prostate[4].asfactor()  # DPROS -> Factor
    prostate[5] = prostate[5].asfactor()  # DCAPS -> Factor
    prostate = prostate.drop('ID')  # remove ID
    prostate.describe()

    params = {'loss': "CrossEntropy", 'hidden': [10, 10], 'use_all_factor_levels': False, 'standardize':True, 'missing_values_handling':'skip', 'sparse':True}
    pyunit_utils.javapredict("deeplearning", "class", prostate, prostate, list(set(prostate.names) - {"CAPSULE"}),
                              "CAPSULE", pojo_model=False, **params) # want to build mojo


# generate random dataset
def random_dataset(response_type, verbose=True):
    """Create and return a random dataset."""
    if verbose: print("\nCreating a dataset for a %s problem:" % response_type)
    fractions = {k + "_fraction": random.random() for k in "real categorical integer time string binary".split()}
    fractions["string_fraction"] = 0  # Right now we are dropping string columns, so no point in having them.
    fractions["binary_fraction"] /= 3
    fractions["time_fraction"] /= 2
    # fractions["categorical_fraction"] = 0
    sum_fractions = sum(fractions.values())
    for k in fractions:
        fractions[k] /= sum_fractions
    response_factors = (1 if response_type == "regression" else
                        2 if response_type == "binomial" else
                        random.randint(10, 30))
    df = h2o.create_frame(rows=random.randint(15000, 25000) + NTESTROWS, cols=random.randint(20, 100),
                          missing_fraction=random.uniform(0, 0.05),
                          has_response=True, response_factors=response_factors, positive_response=True,
                          **fractions)
    if verbose:
        print()
        df.show()
    return df

if __name__ == "__main__":
    pyunit_utils.standalone_test(deeplearning_mojo)
else:
    deeplearning_mojo()
