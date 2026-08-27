# AWS Identity and Access Management (IAM)

IAM is the AWS service that controls **who** (identity) can do **what**
(action) on **which resources**, and under **what conditions**. Every
API call made to AWS — including a Bedrock `invoke_model` request — is
evaluated against IAM before it is allowed to proceed.

## Core concepts

- **Users, groups, and roles.** IAM users represent people or
  applications with long-term credentials; groups collect users to
  apply the same policies to all of them; roles are assumed
  temporarily (by a person, or by an AWS service like Lambda) and
  grant permissions without long-lived credentials.
- **Policies.** Permissions are defined in JSON policy documents that
  list allowed or denied actions (e.g. `bedrock:InvokeModel`) against
  specific resources, optionally restricted by conditions (such as
  source IP or time of day).
- **Identity-based vs. resource-based policies.** Identity-based
  policies attach to a user, group, or role; resource-based policies
  (like an S3 bucket policy) attach to the resource itself and can
  grant access to other accounts.
- **Least privilege.** The standard best practice is to grant only the
  specific actions and resources a workload actually needs — for
  example, a Lambda function that only calls Bedrock and writes logs
  should not also have permission to delete S3 buckets.
- **Temporary credentials.** The AWS Security Token Service (STS)
  issues short-lived credentials when a role is assumed, which is
  safer than distributing permanent access keys.

## How this repository uses it

Every example in this repository — the chatbot, the RAG pipeline, and
the Lambda handler — ultimately runs under IAM permissions: either the
IAM user/role behind your local AWS credentials (`aws configure`,
`~/.aws/credentials`, or `AWS_PROFILE`) when running locally, or the
Lambda execution role defined for the deployed function in
`infrastructure/template.yaml`.
