from mteb_rarb_models.instruct_instructor import InstructionInstructor
import mteb

task_names = ["ARCChallenge","AlphaNLI","HellaSwag","PIQA","Quail","SIQA","WinoGrande","SpartQA",
              "TempReasonL1","TempReasonL2Pure","TempReasonL2Fact","TempReasonL3Pure","TempReasonL3Fact",
              "RARbCode","RARbMath"]

for model_name in ['hkunlp/instructor-base',
                   'hkunlp/instructor-large']:
    print(model_name)
    model = InstructionInstructor(model_name, device = "cuda:0",
                                  do_instruction=False)
    tasks = mteb.get_tasks(tasks=task_names)

    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(model, #batch_size = 16,#score_function = "dot",
                            output_folder=f"results-leaderboard/{model_name}/RAR-b-wo-inst")

for model_name in ['hkunlp/instructor-base',
                   'hkunlp/instructor-large']:
    print(model_name)
    model = InstructionInstructor(model_name, device = "cuda:0",
                                  do_instruction=True, do_doc_instruction=True)
    tasks = mteb.get_tasks(tasks=task_names)

    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(model, #batch_size = 16,#score_function = "dot",
                            output_folder=f"results-leaderboard/{model_name}/RAR-b-w-inst")
    

for model_name in ['hkunlp/instructor-XL']:
    print(model_name)
    model = InstructionInstructor(model_name, device = "cuda:0",
                                  do_instruction=False)
    tasks = mteb.get_tasks(tasks=task_names)

    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(model, batch_size = 16,#score_function = "dot",
                            output_folder=f"results-leaderboard/{model_name}/RAR-b-wo-inst")

for model_name in ['hkunlp/instructor-XL']:
    print(model_name)
    model = InstructionInstructor(model_name, device = "cuda:0",
                                  do_instruction=True, do_doc_instruction=True)
    tasks = mteb.get_tasks(tasks=task_names)

    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(model, batch_size = 16,#score_function = "dot",
                            output_folder=f"results-leaderboard/{model_name}/RAR-b-w-inst")