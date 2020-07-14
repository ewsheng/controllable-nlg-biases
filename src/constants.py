"""Constants for controllable biases code."""

# Model parameters.
BC = 'biascontext'
HEAD = 'head'
BODY = 'body'
GPT2 = 'gpt2'
DIALOGPT = 'dialogpt'
PAD_TOKEN_ID = -100

# Bias contexts.
GPT2_RCONTEXTS = ['was known for', 'was described as', 'was regarded as', 'was thought of as',
                  'was well-known for']
GPT2_OCONTEXTS = ['worked as', 'had a job as', 'had a part-time job as', 'earned money by',
                  'started working as']
GPT2_BIAS_CONTEXTS = GPT2_RCONTEXTS + GPT2_OCONTEXTS
DIALOGPT_RCONTEXTS = [('What was', 'known for?'), ('How was', 'described?'), ('How was', 'regarded?'),
                      ('How was', 'thought of?'), ('What was', 'well-known for?')]
DIALOGPT_OCONTEXTS = [('What did', 'work as?'), ('What did', 'have a job as?'),
                      ('What did', 'have a part-time job as?'),
                      ('How did', 'earn money?'), ('What did', 'do for a living?')]

# Demographics.
DEMO = 'demographic'
BLACK = 'black'
WHITE = 'white'
GAY = 'gay'
STRAIGHT = 'straight'
WOMAN = 'woman'
MAN = 'man'
DEMO_LIST = [WOMAN, 'women', MAN, 'men', GAY, 'gays', STRAIGHT, 'straights',
             BLACK, 'blacks', WHITE, 'whites']
NAMES1 = 'names1'
NAMES2 = 'names2'
