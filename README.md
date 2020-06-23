# Conversational AI Chatbot With AWS Lambda Function and EFS

With the release of Amazon EFS for Lambda, you can now easily share data across function invocations. It also opens new capabilities, such as building and importing large libraries and machine learning models directly into your Lambda functions. Let's go over how to build a serverless conversational AI chatbot using Lambda function and EFS.

This project contains source code and supporting files for a serverless Conversational AI Chatbot. It includes the following files and folders.

- src/api.py - Code for the application's Lambda function.
- template.yaml - A template that defines the application's AWS resources.

The application uses several AWS resources. These resources are defined in the `template.yaml` file in this project. You can update the template to add AWS resources through the same deployment process that updates your application code.

## Deploy the sample application

To build and deploy your application for the first time, run the following in your shell:

```bash
sam build --use-container
sam deploy --guided
```

