#%%
import preprocessing as pp
from claim_verifier import ClaimCheck
from contrarian_claims import contrarian_claims_full
import os
# import nltk
# from nltk.tokenize import sent_tokenize
from frame_extractor import FrameExtract
import re
import pickle
import logging
#%%


newspaper = "telegraph"
verifier = ClaimCheck()
FX = FrameExtract(claims_file=contrarian_claims_full)
input_folder = rf'articles/{newspaper}/txt/'
frame_output_folder = rf'processed_data/{newspaper}/frames/'
claim_output_folder = rf'processed_data/{newspaper}/claims/'
# Loop through all files in the input folder
filename_list = [filename for filename in os.listdir(input_folder) if filename.endswith(".txt")]

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(f'{newspaper}_PIPE.log')
fh.setLevel(logging.DEBUG) # or any level you want
logger.addHandler(fh)

#df_path = f"{newspaper}_claims.csv"
#dataframe = pp.load_or_create_df(df_path)
#%%
# find all articles
for filename in filename_list:
    
    # check if file has already been processed

    # base_filename = filename.replace('.txt', '')
    # if any(re.match(f'{base_filename}_\d+_FRAME\.pkl', f) for f in os.listdir(frame_output_folder)):
    #     continue
    if filename.replace('.txt', '_FRAME.pkl')in os.listdir(frame_output_folder):
        continue
    # load file and get meta data
    text = pp.load_text(f"articles/{newspaper}/txt/{filename}")
    date, length, author, title = pp.extract_metadata(text, filename)
    
    # skip article if it is over 1500 words long.
    if length > 1500:
        continue

    # check that article is about climate change
    question = "Does this TEXT discuss climate change?"
    if verifier.claim_check(text = text, question = question) == False:
        continue

    logger.info("==================================================")
    logger.info("**************************************************")
    logger.info(f"TITLE: {title}")
    logger.info("**************************************************")
    logger.info("==================================================")


    # check if contrarian claims are present in the article

    text_claims = {'claims': [], 'claim_ids': []}

    logger.info("LOOKING FOR CONTRARIAN CLAIMS")
    for claim_id, claim_details in contrarian_claims_full.items():
        claim_description = claim_details["description"]
        question = f"Does the TEXT explicitly or implicitly present, discuss, or relate to the following claim: {claim_description}?"
        match = verifier.claim_check(text = text, question = question)
        if match:
            text_claims['claims'].append(claim_description)
            text_claims['claim_ids'].append(claim_id)

    # generate frames

    logger.info("BEGIN FRAME SUMMARY PROCESS")
    all_frames = {id: {'best_frame': '', 'candidate_frames': []} for id in text_claims['claim_ids']}
    
    for k in range(len(text_claims['claim_ids'])):
        claim_description = text_claims['claims'][k]
        id = text_claims['claim_ids'][k]

        logger.info("GENERATING DETECTION THOUGHTS")
        # generate frame detection wrt claim in article x4
        detective_thoughts = []
        for agent in range(4):
            thought = FX.CoT_detect_framing(article = text, claim = claim_description)
            logger.info(thought)
            detective_thoughts.append(thought)

        # use all detection reasonings to make a final decision - is framing used or no?
        is_framing = FX.hivemind_detection(article = text, claim = claim_description, thought_list=detective_thoughts)
        if is_framing == False:
            logger.info("FRAMING EFFECT NOT DETECTED")
            continue

        logger.info("GENERATING FRAME EXPLANATIONS")
        # generate explanation of how framing is used wrt claim x4
        frame_explanations = []
        for agent in range(4):
            thought = FX.CoT_explain_framing(article = text, claim = claim_description)
            logger.info(thought)
            frame_explanations.append(thought)

        final_explanation = FX.update_explanation(article = text, claim = claim_description, thought_list=frame_explanations)

        logger.info("GENERATING FRAME SUMMARIES")
        # use final explanation to generate a frame summary x4
        candidate_summaries = []
        for agent in range(4):
            thought = FX.generate_frame_summary(article = text, claim = claim_description, explanation=final_explanation)
            logger.info(thought)
            candidate_summaries.append(thought)

        logger.info("CHOOSING BEST FRAME SUMMARY")
        # vote on best summary.
        final_frame = FX.hivemind_choose_best_frame(article = text, claim = claim_description, thought_list = candidate_summaries)
        logger.info(final_frame)

        logger.info("SAVING DATA")
        # save frame summaries
        all_frames[id]['best_frame'] = final_frame
        all_frames[id]['candidate_frames'] = candidate_summaries

    fname = frame_output_folder+filename.replace('.txt', '_FRAME.pkl')
    f = open(fname, 'wb')
    pickle.dump(all_frames, f)
    f.close()
    fname = claim_output_folder + filename.replace('.txt', f'_CLAIM.pkl')
    f = open(fname, 'wb')
    pickle.dump(text_claims, f)
    f.close()

