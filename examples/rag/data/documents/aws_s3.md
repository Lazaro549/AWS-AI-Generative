# Amazon S3

Amazon Simple Storage Service (S3) is AWS's object storage service. It
stores data as objects (files plus metadata) inside buckets, addressed
by a key (essentially a path-like string), rather than as blocks or
files on a traditional filesystem.

## Core concepts

- **Buckets and keys.** A bucket is a top-level, globally-uniquely-named
  container; objects inside it are identified by their key. There is
  no real directory structure — key prefixes (e.g. `documents/aws.md`)
  are used to simulate folders in tools and the console.
- **Durability and availability.** S3 is designed for very high
  durability by storing redundant copies of each object across
  multiple facilities within a region.
- **Storage classes.** Objects can live in different storage classes
  (Standard, Infrequent Access, Glacier, and others) that trade
  retrieval speed for cost, and lifecycle rules can move objects
  between classes automatically as they age.
- **Versioning.** When enabled on a bucket, every write to a key
  creates a new version instead of overwriting the previous one,
  which protects against accidental deletes or overwrites.
- **Access control.** Access is governed by IAM identity-based
  policies, S3 bucket policies (resource-based), and — for
  fine-grained per-object control — access control lists (ACLs),
  which AWS now recommends avoiding in favor of policies. Presigned
  URLs allow granting temporary, scoped access to a specific object
  without making it public.

## Typical use cases

S3 is commonly used for static website hosting, data lake storage,
backups and archives, and as the durable storage layer behind
analytics and machine learning pipelines. In generative AI
applications, S3 is a common place to store a raw document corpus
before it is loaded, chunked, and embedded for retrieval — the
same role that `examples/rag/data/` plays locally in this repository.
