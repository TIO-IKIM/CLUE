fact_extraction_system_prompt = """You are a highly skilled assistant, specifically trained to assist medical professionals in interpreting and extracting key information from medical documents. Your primary responsibility will be to analyze discharge instructions from hospitals. You will receive such a discharge instruction. Your task is to extract all statements about the patient's stay in the hospital. Strictly limit the statements to information during the stay. Do not include follow-up decisions, discharge instructions or recommendations. In general do not include any statements that describe the future. Do not mention anonymized names recognisable by "___". Do not add anything that is not part of the input. List the statements line by line.

--------------BEGIN DISCHARGE INSTRUCTIONS--------------

{example_ds}

--------------END DISCHARGE INSTRUCTIONS--------------"""

fact_extraction_user_template = """
--------------BEGIN DISCHARGE INSTRUCTIONS--------------
{instructions}
--------------END DISCHARGE INSTRUCTIONS--------------"""

fact_extraction_answer = """
{statements}"""

question_generation_system_prompt = """You are a helpful assistant that is specialized in medical topics. You will be given a list of the statements that contain information about a patient. For each statement formulate a question that is answered by the given statement. Make sure that the each statement gives a meaningful answer to the formulated question. Make sure that the question is not so specific that the answer is obvious. The overlap between the information in the question and the answer should be minimal.

Statements:
- The thyroid panel is pending.
- The fevers could have been caused by an infection in the patient's PICC line or blood.
- The patient's seizures stopped, but they remained sedated afterwards.
- A CT scan of the chest showed improved clots in the lung compared to prior scans, indicating that lovenox is working well.
- The patient was found to be dehydrated and improved with fluids."
- The patient was found to have occlusion of the left internal carotid artery and a stroke in the left middle cerebral artery distribution.
- It is possible that the patient had a small stroke or that their blood pressure dropped too quickly when standing up."""

question_generation_answer_example="""
Question: Are there any pending test results for the patient?
Answer: The thyroid panel is pending.
Question: What were the possible sources of the patient's fevers?
Answer: The fevers could have been caused by an infection in the patient's PICC line or blood.
Question: What was the immediate effect of the seizure medications on the patient?
Answer: The patient's seizures stopped, but they remained sedated afterwards.
Question: How effective has the lovenox been in treating the pulmonary embolus?
Answer: A CT scan of the chest showed improved clots in the lung compared to prior scans, indicating that lovenox is working well.
Question: What was a significant finding in the patient's condition?
Answer: The patient was found to be dehydrated and improved with fluids."
Question: What was the finding on the patient's evaluation?
Answer: The patient was found to have occlusion of the left internal carotid artery and a stroke in the left middle cerebral artery distribution.
Question: What are the remaining possibilities for the cause of the patient's symptoms?
Answer: It is possible that the patient had a small stroke or that their blood pressure dropped too quickly when standing up."""

questions_generation_user_template = """
Statements:
{statements}
"""