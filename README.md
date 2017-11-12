# A2C
PyTorch implementation of Advantage Actor-Critic (A2C)

## Usage

Example command line usage:
````
python main.py BreakoutDeterministic-v3 --num-workers 8 --render
````
This will train the agent on BreakoutDeterministic-v3 with 8 parallel environments, and render each environment. Rendering Gym environments in subprocesses is only supported on Linux as of November 2017.

## References
[OpenAI/baselines](https://github.com/openai/baselines)

[OpenAI/universe-starter-agent](https://github.com/openai/universe-starter-agent)

[ikostrikov/pytorch-a3c](https://github.com/ikostrikov/pytorch-a3c)
