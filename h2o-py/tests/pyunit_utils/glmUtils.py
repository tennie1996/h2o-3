from __future__ import print_function
from future import standard_library
standard_library.install_aliases()
from builtins import range
from past.builtins import basestring
from functools import reduce
from scipy.sparse import csr_matrix
import sys, os, gc
import pandas as pd
import tempfile
from h2o.estimators.glm import H2OGeneralizedLinearEstimator

# This file will contain functions used by GLM test only.
def assertEqualRegPaths(keys, pathList, index, onePath, tol=1e-6):
    for oneKey in keys:
        if (pathList[oneKey] != None):
            assert abs(pathList[oneKey][index]-onePath[oneKey][0]) < tol, \
                "Expected value: {0}, Actual: {1}".format(pathList[oneKey][index], onePath[oneKey][0])



def assertEqualCoeffDicts(coef1Dict, coef2Dict, tol = 1e-6):
    assert len(coef1Dict) == len(coef2Dict), "Length of first coefficient dict: {0}, length of second coefficient " \
                                             "dict: {1} and they are different.".format(len(coef1Dict, len(coef2Dict)))
    for key in coef1Dict:
        assert abs(coef1Dict[key]-coef2Dict[key]) < tol, "Coefficient for {0} from first dict: {1}, from second dict:" \
                                                         " {2} and they are different.".format(key, coef1Dict[key],
                                                                                               coef2Dict[key])
def assertEqualModelMetrics(metrics1, metrics2, tol = 1e-6,
                            keySet=["MSE", "AUC", "Gini", "null_deviance", "logloss", "RMSE",
                                    "pr_auc", "r2"]):
    # 1. Check model types
    model1_type = metrics1.__class__.__name__
    model2_type = metrics2.__class__.__name__
    assert model1_type is model2_type, "The model types differ. The first model metric is of type {0} and the second " \
                                       "model metric is of type {1}.".format(model1_type, model2_type)

    metricDict1 = metrics1._metric_json
    metricDict2 = metrics2._metric_json

    for key in keySet:
        if key in metricDict1.keys() and (isinstance(metricDict1[key], float)): # only compare floating point metrics
            assert abs(metricDict1[key]-metricDict2[key])/max(1,max(metricDict1[key],metricDict2[key])) < tol, \
                "ModelMetric {0} from model 1,  {1} from model 2 are different.".format(metricDict1[key],metricDict2[key])

# When an array of alpha and/or lambdas are given, a list of submodels are also built.  For each submodel built, only
# the coefficients, lambda/alpha/deviance values are returned.  The model metrics is calculated from the submodel
# with the best deviance.  
#
# In this test, in addition, we build separate models using just one lambda and one alpha values as when building one
# submodel.  In theory, the coefficients obtained from the separate models should equal to the submodels.  We check 
# and compare the followings:
# 1. coefficients from submodels and individual model should match when they are using the same alpha/lambda value;
# 2. training metrics from alpha array should equal to the individual model matching the alpha/lambda value;
def compareSubmodelsNindividualModels(modelWithArray, trainingData, xarray, yindex):
    best_submodel_index = modelWithArray._model_json["output"]["best_submodel_index"]
    r = H2OGeneralizedLinearEstimator.getGLMRegularizationPath(modelWithArray)  # contains all lambda/alpha values of submodels trained.
    submodel_num = len(r["lambdas"])
    regKeys = ["alphas", "lambdas", "explained_deviance_valid", "explained_deviance_train"]
    for submodIndx in range(submodel_num):  # manually build glm model and compare to those built before
        modelGLM = H2OGeneralizedLinearEstimator(family='binomial', alpha=[r["alphas"][submodIndx]], Lambda=[r["lambdas"][submodIndx]])
        modelGLM.train(training_frame=trainingData, x=xarray, y=yindex)
        # check coefficients between submodels and model trained with same parameters
        assertEqualCoeffDicts(r["coefficients"][submodIndx], modelGLM.coef())
        modelGLMr = H2OGeneralizedLinearEstimator.getGLMRegularizationPath(modelGLM) # contains one item only
        assertEqualRegPaths(regKeys, r, submodIndx, modelGLMr)
        if (best_submodel_index == submodIndx):  # check training metrics of modelGLM should equal that of m since it is the best subModel
            assertEqualModelMetrics(modelWithArray._model_json["output"]["training_metrics"],
                                    modelGLM._model_json["output"]["training_metrics"])
            assertEqualCoeffDicts(modelWithArray.coef(), modelGLM.coef()) # model coefficient should come from best submodel
        else:  # check and make sure best_submodel_index has lowest deviance
            assert modelGLM.residual_deviance() - modelWithArray.residual_deviance() >= 0, \
                "Individual model has better residual_deviance than best submodel!"

def extractNextCoeff(cs_norm, orderedCoeffNames, startVal):
    for ind in range(0, len(startVal)):
        startVal[ind] = cs_norm[orderedCoeffNames[ind]]
    return startVal

def assertCoefEqual(regCoeff, coeff, coeffClassSet, tol=1e-6):
    for key in regCoeff:
        temp = key.split('_')
        classInd = int(temp[1])
        val1 = regCoeff[key]
        val2 = coeff[coeffClassSet[classInd]][temp[0]]
        assert type(val1)==type(val2), "type of coeff1: {0}, type of coeff2: {1}".format(type(val1), type(val2))
        diff = abs(val1-val2)
        print("val1: {0}, val2: {1}, tol: {2}".format(val1, val2, tol))
        assert diff < tol, "diff {0} exceeds tolerance {1}.".format(diff, tol)
