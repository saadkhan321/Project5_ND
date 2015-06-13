--------------------------------------------------------------------------------------------------------------------------

VERSION 2
--------------------------------------------------------------------------------------------------------------------------

Name of the Final Algorithm python file is still poi_id.py



Following additional material is included in the project submission V2

*******************
Section 4
*******************

This section was re-structured in the following manner:

- Notes have been added to sub-section 4.1 to explain the thought process for feature selection for initial algorithm performance evaluation to identify the best performing algorithm

- Section 4.1 now covers only initial feature selection for comparative algorithm analysis instead of final algorithm feature list

- Section 4.4 was created to accommodate the details of feature scaling

- The previous statement regarding feature scaling is now removed and new section (section 4.4) added with more details

*******************
Section 6
*******************

Section 6 was re-structured to address the better representation of the definition of precision and recall and its application in this project when applied to POIs.

- The Predicted Class/Actual Class table was re-drawn as per the reviewer's comments in order to avoid confusion and correctly define precision and recall.

- Definition of precision was corrected as per the reviewer's comments.


*******************
Section 8
*******************

A completely new section, section 8 was created in order to explain the flow of feature selection for the final algorithm implementation.

- Performance Metrics + feature importances were documented for the original feature list

- Based on this, new filtered feature list was selected

- Performance Metrics + feature importances were documented for the filtered feature list

- The final feature list to be used in the final implementation fo the algorithm was defined


*******************
References
*******************

New resources consulted for the updated project have been appended to the reference list at the end of the project report



*******************
Additional Ipython Notebook
*******************

An additional ipython notebook: 'FinalProject_WriteUp_Algo_POI_ID_Test.ipynb' is added the final_project folder, this includes all the working done to updated the project report.




--------------------------------------------------------------------------------------------------------------------------

VERSION 1
--------------------------------------------------------------------------------------------------------------------------

1. Project Report: Includes answers to the free-response questions + references + walk through of the analysis done in the project.

2. Ouput Pickle Files: 
			
	- my_dataset.pkl, 
	- my_classifier.pkl, 
	- my_feature_list.pkl

3. poi_id.py: Implementation code for the final algorithm used in the analysis +  outliers removal/handling + new feature creation.

4. tester.py: This will be used  fro testing the POI identifier.

5. Ipython Notebook for all 9 scenarios of features settings tested for each of the algorithms:

	- FinalProject_WriteUp_Algo_Test_OriginalFeatures.ipynb
	- FinalProject_WriteUp_Algo_Test_OriginalFeatures.ipynb
	- FinalProject_WriteUp_Algo_Test_MoreSelectedOriginalFeatures.ipynb
	- FinalProject_WriteUp_Algo_Test_FractionFeatures_OriginalFeatures.ipynb
	- FinalProject_WriteUp_Algo_Test_FractionFeatures_SelectedOriginalFeatures.ipynb
	- FinalProject_WriteUp_Algo_Test_TextFeatures_OriginalFeatures.ipynb
	- FinalProject_WriteUp_Algo_Test_TextFeatures_SelectedOriginalFeatures.ipynb
	- FinalProject_WriteUp_Algo_Test_AllNewFeatures_OriginalFeatures.ipynb
	- FinalProject_WriteUp_Algo_Test_AllNewFeatures_SelectedOriginalFeatures.ipynb

NOTE: A list of Web sites, books, forums, blog posts and github repositories that were referred for this project are included in the reference section at the end of the Project Report.
