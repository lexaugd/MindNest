## High-Level Requirements of the Interface Approach


This section focuses on the high-level requirements of the interface
approach that are needed to implement the reference architecture
depicted in @fig:arch.

### Technology- and Vendor-Agnostic

Due to the many different tools, services, and infrastructures
available in the general area of Big Data, an interface ought to be as
vendor-independent as possible, while, at the same time, be able to
leverage best practices. Hence, a methodology is needed that allows
extension of interfaces to adapt and leverage existing approaches, but
also allows the interfaces to provide merit in easy specifications
that assist the formulation and definition of the NBDRA.

### Support of Plug-In Compute Infrastructure

As Big Data is not just about hosting data, but about analyzing data,
the interfaces provided herein must encapsulate a rich infrastructure
environment that is used by data scientists. This includes the ability
to integrate (or plug-in) various compute resources and services to
provide the necessary compute power to analyze the data. These
resources and services include the following:

* Access to hierarchy of compute resources from the laptop/desktop,
  servers, data clusters, and clouds;
* The ability to integrate special-purpose hardware such as graphics
  processing units (GPUs) and field-programmable gate arrays (FPGAs)
  that are used in accelerated analysis of data; and
* The integration of services including microservices that allow the
  analysis of the data by delegating them to hosted or dynamically
  deployed services on the infrastructure of choice.

### Orchestration of Infrastructure and Services

From review of the use case collection, presented in the *NBDIF:
Volume 3, Use Cases and General Requirements* document [@www-vol3-v3],
the need arose to address the mechanism of preparing suitable
infrastructures for various use cases. As not every infrastructure is
suited for every use case, a custom infrastructure may be needed. As
such, this document is not attempting to deliver a single deployed
NBDRA, but allow the setup of an infrastructure that satisfies the
particular use case. To achieve this task, it is necessary to
provision software stacks and services while orchestrating their
deployment and leveraging infrastructures. It is not the focus of this
document to replace existing orchestration software and services, but
provide an interface to them to leverage them as part of defining and
creating the infrastructure. Various orchestration frameworks and
services could therefore be leveraged, even as part of the same
framework, and work in orchestrated fashion to achieve the goal of
preparing an infrastructure suitable for one or more applications.

### Orchestration of Big Data Applications and Experiments

The creation of the infrastructure suitable for Big Data applications
provides the basic computing environment. However, Big Data applications
may require the creation of sophisticated applications as part of
interactive experiments to analyze and probe the data. For this purpose,
the applications must be able to orchestrate and interact with
experiments conducted on the data while assuring reproducibility and
correctness of the data. For this purpose, a *System Orchestrator*
(either the data scientists or a service acting on behalf of the data
scientist) is used as the command center to interact on behalf of the
Big Data Application Provider to orchestrate dataflow from Data
Provider, carry out the Big Data application life cycle with the help of
the Big Data Framework Provider, and enable the Data Consumer to consume
Big Data processing results. An interface is needed to describe these
interactions and to allow leveraging of experiment management frameworks
in scripted fashion. A customization of parameters is needed on several
levels. On the highest level, application-motivated
parameters are needed to drive the orchestration of the experiment. On
lower levels, these high-level parameters may drive and create
service-level agreements, augmented specifications, and parameters that
could even lead to the orchestration of infrastructure and services to
satisfy experiment needs.

### Reusability

The interfaces provided must encourage reusability of the
infrastructure, services, and experiments described by them. This
includes (1) reusability of available analytics packages and services
for adoption; (2) deployment of customizable analytics tools and
services; and (3) operational adjustments that allow the services and
infrastructure to be adapted while at the same time allowing for
reproducible experiment execution.

### Execution Workloads

One of the important aspects of distributed Big Data services can be
that the data served is simply too big to be moved to a different
location. Instead, an interface could allow the description and
packaging of analytics algorithms, and potentially also tools, as a
payload to a data service. This can be best achieved, not by sending the
detailed execution, but by sending an interface description that
describes how such an algorithm or tool can be created on the server and
be executed under security considerations (integrated with
authentication and authorization in mind).

### Security and Privacy Fabric Requirements

Although the focus of this document is not security and privacy, which
are documented in the *NBDIF: Volume 4, Security and Privacy*
[@www-vol4-v3], the interfaces defined herein must be capable of
integration into a secure reference architecture that supports secure
execution, secure data transfer, and privacy. Consequently, the
interfaces defined herein can be augmented with frameworks and
solutions that provide such mechanisms.  Thus, diverse requirement
needs stemming from different use cases addressing security need to be
distinguished. To contrast that the security requirements between
applications can vary drastically, the following example is
provided. Although many of the interfaces and their objects to support
Big Data applications in physics are similar to those in healthcare,
they differ in the integration of security interfaces and
policies. While in physics the protection of data is less of an issue,
it is a stringent requirement in healthcare. Thus, deriving
architectural frameworks for both may use largely similar components,
but addressing security issues will be very different. The security of
interfaces may be addressed in other documents.  In this document,
they are considered an advanced use case showcasing that the validity
of the specifications introduced here is preserved, even if security
and privacy requirements differ vastly among application use cases.

