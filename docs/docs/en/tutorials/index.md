# Tutorials

In this series of tutorials, you'll learn how to use the **FastAgency** framework to create interactive chatbots that can scrape the web and work with different APIs to respond to user requests.

---

## Available Tutorials

### 1. Web Scraping and Giphy API Integration

In this tutorial, we will explore how to leverage the **FastAgency** framework to create a dynamic chatbot that integrates two powerful agents:

- [**`WebSurferAgent`**](../api/fastagency/runtimes/autogen/agents/websurfer/WebSurferAgent.md) – A web-scraping agent capable of retrieving relevant content from webpages.
- **Giphy agent** – An agent that interacts with the [Giphy API](https://giphy.com){target="_blank"} to fetch GIFs based on the user’s request.

### When to Use Web Scraping and Giphy API Integration?
- **API Integration**: Learn how to integrate external APIs like Giphy with FastAgency.
- **Autonomous Agents**: Build and register agents that autonomously scrape the web for relevant information.
- **User-Agent Workflows**: Use [AutoGen workflows](../api/fastagency/runtimes/autogen/AutoGenWorkflows.md) to manage agent interactions and user input.
- **Personalized Content**: Present scraped content to users and offer personalized GIF suggestions based on that content.

[Let’s dive into Web Scraping and Giphy API Integration →](giphy/index.md)

---

### 2. WhatsApp API Integration and Web Scraping

In this tutorial, we will explore how to build a chatbot using the **FastAgency** framework that integrates two essential agents:

- [**`WebSurferAgent`**](../api/fastagency/runtimes/autogen/agents/websurfer/WebSurferAgent.md) – A web-scraping agent capable of retrieving content from websites.
- **WhatsApp agent** – An agent that interacts with the [Infobip WhatsApp API](https://www.infobip.com/docs/api/channels/whatsapp){target="_blank"} to send messages via WhatsApp.

### When to Use WhatsApp API Integration and Web Scraping?
- **API Integration**: Learn how to integrate the Infobip WhatsApp API using OpenAPI.
- **Autonomous Agents**: Build and register agents that autonomously scrape the web using the [**`WebSurferAgent`**](../api/fastagency/runtimes/autogen/agents/websurfer/WebSurferAgent.md).
- **User-Agent Workflows**: Manage user interactions and send scraped content via WhatsApp using [AutoGen workflows](../api/fastagency/runtimes/autogen/AutoGenWorkflows.md).
- **Security**: Handle secure API credentials using [**`APIKeyHeader`**](../api/fastagency/api/openapi/security/APIKeyHeader.md) and ensure safe communication.

[Let’s dive into WhatsApp API Integration and Web Scraping →](whatsapp/index.md)
