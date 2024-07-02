from mteb.models.instructions import task_to_instruction
from rarb_model.instruct_st import InstructionSentenceTransformer
import mteb

task_names = ["ARCChallenge","AlphaNLI","HellaSwag","PIQA","Quail","SIQA","WinoGrande","SpartQA",
              "TempReasonL1","TempReasonL2Pure","TempReasonL2Fact","TempReasonL3Pure","TempReasonL3Fact",
              "RARbCode","RARbMath"]

model_names = ["sentence-transformers/all-mpnet-base-v2",
              "facebook/contriever",
              "BAAI/bge-small-en-v1.5",
              "BAAI/bge-base-en-v1.5",
              "BAAI/bge-large-en-v1.5"
              "BAAI/bge-m3",
              ]

# without instruction
for model_name in model_names:
    model = InstructionSentenceTransformer(model_name,
                                        do_instruction = False)

    tasks = mteb.get_tasks(tasks=task_names)

    evaluation = mteb.MTEB(tasks=tasks)
    if "contriever" in model_name:
        results = evaluation.run(model, score_function = "dot",
                                output_folder=f"results-leaderboard/{model_name}/RAR-b-wo-inst")
    else:
        results = evaluation.run(model, #score_function = "dot",
                                output_folder=f"results-leaderboard/{model_name}/RAR-b-wo-inst")

# with instruction
for model_name in model_names:
    model = InstructionSentenceTransformer(model_name,
                                        do_instruction = True)

    tasks = mteb.get_tasks(tasks=task_names)

    evaluation = mteb.MTEB(tasks=tasks)
    if "contriever" in model_name:
        results = evaluation.run(model, score_function = "dot",
                                output_folder=f"results-leaderboard/{model_name}/RAR-b-w-inst")
    else:
        results = evaluation.run(model, #score_function = "dot",
                                output_folder=f"results-leaderboard/{model_name}/RAR-b-w-inst")