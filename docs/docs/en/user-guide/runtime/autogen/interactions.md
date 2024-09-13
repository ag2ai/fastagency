# Custom User Interactions

In this example, we'll demonstrate how to create custom interaction with the user using [`UI`](../../../../api/fastagency/UI/) protocol and its [`process_message`](../../../../api/fastagency/UI/#fastagency.UI.create_subconversation) method.


## Install

To get started, you need to install FastAgency. You can do this using `pip`, Python's package installer.

```console
pip install "fastagency[autogen]"
```


## Define Interaction

This section describes how to define functions for the `ConversableAgent` instances representing the student and teacher. We will also explain the differences between `MultipleChoice`, `SystemMessage`, and `TextInput`, which are used for communication between the user and agents.

Let's define three functions which will be available to the agents:

### Free Textual Tnput

`TextInput` is suitable for free-form text messages, ideal for open-ended queries and dialogues. This function allows the student to request exam questions from the teacher and provides some suggestions using `TextInput`.

```python
{! docs_src/user_guide/custom_user_interactions/main.py [ln:52.5,53.5,54.5,55.5,56.5,57.5,58.5,59.5,60.5,61.5,62.5,63.5,64.5,66.5,67.5,68.5,69.5,70.5] !}
```

### System Info Messages

`SystemMessage` is used for operational or system-related instructions, such as logging data, and is not part of the agent dialogue. This function logs the final answers after the student completes the discussion using `SystemMessage` to log the event.

```python
{! docs_src/user_guide/custom_user_interactions/main.py [ln:72.5,73.5,74.5,75.5,76.5,77.5,78.5,79.5,80.5,81.5,82.5,83.5,84.5,85.5] !}
```

### Multiple Choice

`MultipleChoice` is used for structured responses where the user must select one of several predefined options. This function retrieves the final grade for the student's submitted answers using `MultipleChoice`, presenting the user with grading options.

```python
{! docs_src/user_guide/custom_user_interactions/main.py [ln:87.5,88.5,89.5,90.5,91.5,92.5,93.5,94.5,96.5,97.5,98.5,99.5] !}
```

### Other Types of Messages

All supported messages are subclasses of the [IOMessage](../../../../api/fastagency/IOMessage/) base class.

## Registering the Functions
We now register these functions with the workflow, linking the `student_agent` as the caller and the `teacher_agent` as the executor.

```python
{! docs_src/user_guide/custom_user_interactions/main.py [ln:101.5,102.5,103.5,104.5,105.5,106.5,107.5,108.5,109.5,110.5,111.5,112.5,113.5,114.5,115.5,116.5,117.5,118.5,119.5,120.5,121.5,122.5,123.5] !}
```

## Define FastAgency Application
Finally, we'll define the entire application:

```python
{! docs_src/user_guide/custom_user_interactions/main.py!}
```

## Run Application

Once everything is set up, you can run your FastAgency application using the following command:

```console
fastagency run
```
