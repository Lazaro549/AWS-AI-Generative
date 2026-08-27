# AWS Lambda

AWS Lambda is a serverless, event-driven compute service. You upload a
function's code, and Lambda runs it in response to triggers — an HTTP
request through API Gateway, a new object in an S3 bucket, a message on
an SQS queue, a scheduled CloudWatch Events rule, and many others —
without you provisioning or managing any servers.

## Execution model

- **Pay-per-use.** Billing is based on the number of invocations and
  the compute time actually consumed (rounded to the millisecond),
  not on idle capacity.
- **Automatic scaling.** Lambda creates additional execution
  environments in response to concurrent invocations, up to your
  account's concurrency limits, and scales back down to zero when
  there is no traffic.
- **Cold starts.** The first invocation of a new execution environment
  incurs extra latency while the runtime initializes; subsequent
  invocations against a "warm" environment are faster. Memory
  allocation, package size, and runtime choice all affect cold-start
  latency.
- **Execution role.** Every function runs with an IAM role (its
  "execution role") that grants it permission to call other AWS
  services — for example, invoking Bedrock, reading from S3, or
  writing logs to CloudWatch. The function's code can never do more
  than that role allows.
- **Configurable limits.** Memory (which also scales available CPU),
  timeout, and environment variables are all configured per function.

## Typical use cases

Lambda is a natural fit for lightweight APIs, event-driven data
processing, scheduled jobs, and gluing AWS services together. In
generative AI applications, it is frequently used to expose a model
call — such as an Amazon Bedrock `invoke_model` request — behind a
simple HTTP API.

## How this repository uses it

`src/lambda/handler.py` wraps a Bedrock call in a Lambda function so
that the same kind of request made by `examples/chatbot/app.py` can
also be served as an API, deployed through the AWS SAM template in
`infrastructure/template.yaml`.
