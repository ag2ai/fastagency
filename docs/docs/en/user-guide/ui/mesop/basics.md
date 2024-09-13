# Mesop

**[MesopUI](../../../../api/fastagency/ui/mesop/MesopUI/)** in FastAgency offers a web-based interface for interacting with multi-agent workflows. Unlike the **ConsoleUI**, which is text-based and runs in the command line, MesopUI provides a user-friendly browser interface, making it ideal for applications that need a more engaging, graphical interaction. MesopUI is perfect for building interactive web applications and enabling users to interact with agents in a more intuitive way.

To install **FastAgency** with MesopUI support, use the following command:

```bash
pip install "fastagency[autogen,mesop]"
```

This command ensures that the required dependencies for both **AutoGen** and **MesopUI** are installed.

Below, we’ll demonstrate how to set up a basic student-teacher conversation using **[MesopUI](../../../../api/fastagency/ui/mesop/MesopUI/)**.

## Example: Student and Teacher Learning Chat

This example shows how to create a simple learning chat where a student agent interacts with a teacher agent. The student asks questions, and the teacher provides responses, simulating a learning environment. The conversation is facilitated through the web interface using **[MesopUI](../../../../api/fastagency/ui/mesop/MesopUI/)**.

### Step-by-Step Breakdown

#### 1. **Import Required Modules**
We begin by importing the necessary modules from **FastAgency** and **AutoGen**. These imports provide the essential building blocks for creating agents, workflows, and integrating MesopUI.

```python
{! docs_src/getting_started/main_mesop.py [ln:1-8] !}
```

- **ConversableAgent**: This class allows the creation of agents that can engage in conversational tasks.
- **[FastAgency](../../../../api/fastagency/FastAgency/)**: The core class responsible for orchestrating workflows and connecting them with UIs.
- **[UI](../../../../api/fastagency/UI/)** and **[MesopUI](../../../../api/fastagency/ui/mesop/MesopUI/)**: These classes define the user interface for interaction, with **MesopUI** enabling a web-based interaction.
- **[AutoGenWorkflows](../../../../api/fastagency/runtime/autogen/base/AutoGenWorkflows/)**: Manages the creation and execution of multi-agent workflows.

#### 2. **Configure the Language Model (LLM)**
Next, we configure the language model that powers the agents. In this case, we're using **GPT-4o-mini**, and the API key is retrieved from the environment.

```python
{! docs_src/getting_started/main_mesop.py [ln:9-19] !}
```

- **Explanation**: The configuration specifies the LLM model and API key used for powering the conversation between agents. The temperature is set to `0.0` to ensure deterministic responses from the agents, making interactions consistent and reliable.

#### 3. **Define the Workflow and Agents**
Here, we define a simple workflow where the **Student Agent** interacts with the **Teacher Agent**. The student asks questions, and the teacher responds as a math teacher. The workflow is registered using **AutoGenWorkflows**.

```python
{! docs_src/getting_started/main_mesop.py [ln:20-44] !}
```

- **Agent Overview**: The **Student Agent** is configured with a system message, "You are a student willing to learn," and will initiate questions during the interaction. The **Teacher Agent**, on the other hand, is set up as a math teacher and will respond to the student's questions.
- **Workflow Registration**: The workflow is registered under the name `simple_learning`. The **ConversableAgent** class is used to represent both the student and teacher agents, allowing them to communicate with each other up to 5 turns before summarizing the conversation using the `reflection_with_llm` method.

#### 4. **Using MesopUI**
Finally, we instantiate **[MesopUI](../../../../api/fastagency/ui/mesop/MesopUI/)** to link the workflow to a web-based interface. This allows the user to interact with the agents through a web browser.

```python
from fastagency.ui.mesop import MesopUI
app = FastAgency(wf=wf, ui=MesopUI())
```

- **Explanation**: Here, we set up the **MesopUI** as the user interface for the workflow, which will allow the entire agent interaction to take place through a web-based platform.

### Running the Application

Once the workflow is set up, you can run the application using the **FastAgency CLI**. Navigate to the directory where the script is located and run the following command:

```bash
cd docs/docs_src/getting_started
fastagency run main_mesop.py
```

This will launch a local web server, and you will be able to access the MesopUI interface through your browser. The web interface will display the interaction between the student and teacher agents, allowing you to input questions and see the teacher’s responses.

**Note**: Ensure that your OpenAI API key is set in the environment, as the agents rely on it to interact using GPT-4o-mini. If the API key is not correctly configured, the application may fail to retrieve LLM-powered responses.

### Debugging Tips
If you encounter issues running the application, ensure that:
- The OpenAI API key is correctly set in your environment variables.
- All necessary packages are installed, especially the `fastagency[autogen,mesop]` dependencies.
- The MesopUI web interface is accessible from the browser, and no firewall is blocking the connection.

---

By using **[MesopUI](../../../../api/fastagency/ui/mesop/MesopUI/)**, developers can create interactive, web-based multi-agent applications with ease. This interface is ideal for building user-friendly, browser-accessible systems, enabling users to interact with agents in a more engaging and visual environment. You can extend this workflow for more complex scenarios, such as tutoring systems, customer support, or real-time information retrieval.
