# AWS Serverless Application Model (SAM)

AWS SAM is an open-source framework for building serverless
applications on AWS. It is an extension of AWS CloudFormation with a
simplified syntax for defining serverless resources — Lambda
functions, API Gateway APIs, DynamoDB tables, and their event sources
— in a single `template.yaml` file.

## Why SAM instead of raw CloudFormation

A raw CloudFormation template for a Lambda function backed by an API
Gateway endpoint requires defining the function, an API, a deployment,
a stage, and the permissions connecting them individually. SAM
introduces higher-level resource types, such as `AWS::Serverless::
Function`, that expand into all of that boilerplate at deployment
time, so the template stays focused on what the application actually
needs.

## Core workflow

- **`sam build`** resolves dependencies and packages function code
  (and any layers) into a deployable artifact.
- **`sam deploy`** packages the built artifact, uploads it, and creates
  or updates the underlying CloudFormation stack.
- **`sam local`** can invoke a function or start a local API Gateway
  emulator on your machine, so you can test event handling before
  deploying.
- **Guided deploys** (`sam deploy --guided`) walk through stack
  parameters interactively the first time, then remember them for
  subsequent deploys.

## How this repository uses it

`infrastructure/template.yaml` is the SAM template that deploys
`src/lambda/handler.py` as a Lambda function behind an API endpoint, so
the Bedrock-backed handler can be called over HTTP instead of only
locally. The repository's `deploy` GitHub Actions workflow is where
that template is built and deployed automatically.
