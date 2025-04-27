## Component-Specific Interface Requirements


This section summarizes the requirements for the interfaces of the
NBDRA components. The five components are listed in @fig:arch and
addressed in @sec:system-orchestrator-requirements (System
Orchestrator Interface Requirements) and
@sec:data-application-requirements (Big Data Application Provider to
Big Data Framework Provider Interface) of this document. The five main
functional components of the NBDRA represent the different technical
roles within a Big Data system and are the following:

* System Orchestrator: Defines and integrates the required data
  application activities into an operational vertical system (see
  @sec:system-orchestrator-requirements);
* Data Provider: Introduces new data or information feeds into the Big
  Data system (see @sec:data-provider-requirements);
* Data Consumer: Includes end users or other systems that use the
  results of the Big Data Application Provider (see @sec:data-consumer-requirements);
* Big Data Application Provider: Executes a data life cycle to meet
  security and privacy requirements as well as System
  Orchestrator-defined requirements (see @sec:data-application-requirements);
* Big Data Framework Provider: Establishes a computing framework in
  which to execute certain transformation applications while
  protecting the privacy and integrity of data (see @sec:provider-requirements);
  and
* Big Data Application Provider to Framework Provider Interface:
  Defines an interface between the application specification and the
  provider (see @sec:app-provider-requirements).

### System Orchestrator Interface Requirements {#sec:system-orchestrator-requirements}

The System Orchestrator role includes defining and integrating the
required data application activities into an operational vertical
system. Typically, the System Orchestrator involves a collection of more
specific roles, performed by one or more actors, which manage and
orchestrate the operation of the Big Data system. These actors may be
human components, software components, or some combination of the two.
The function of the System Orchestrator is to configure and manage the
other components of the Big Data architecture to implement one or more
workloads that the architecture is designed to execute. The workloads
managed by the System Orchestrator may be assigning/provisioning
framework components to individual physical or virtual nodes at the
lower level, or providing a graphical user interface that supports the
specification of workflows linking together multiple applications and
components at the higher level. The System Orchestrator may also,
through the Management Fabric, monitor the workloads and system to
confirm that specific quality of service requirements is met for each
workload, and may elastically assign and provision additional physical
or virtual resources to meet workload requirements resulting from
changes/surges in the data or number of users/transactions. The
interface to the System Orchestrator must be capable of specifying the
task of orchestration the deployment, configuration, and the execution
of applications within the NBDRA. A simple vendor-neutral specification
to coordinate the various parts either as simple parallel language tasks
or as a workflow specification is needed to facilitate the overall
coordination. Integration of existing tools and services into the System
Orchestrator as extensible interfaces is desirable.

### Data Provider Interface Requirements {#sec:data-provider-requirements}

The Data Provider role introduces new data or information feeds into the
Big Data system for discovery, access, and transformation by the Big
Data system. New data feeds are distinct from the data already in use by
the system and residing in the various system repositories. Similar
technologies can be used to access both new data feeds and existing
data. The Data Provider actors can be anything from a sensor, to a human
inputting data manually, to another Big Data system. Interfaces for data
providers must be able to specify a data provider so it can be located
by a data consumer. It also must include enough details to identify the
services offered so they can be pragmatically reused by consumers.
Interfaces to describe pipes and filters must be addressed.

### Data Consumer Interface Requirements {#sec:data-consumer-requirements}

Like the Data Provider, the role of Data Consumer within the NBDRA can
be an actual end user or another system. In many ways, this role is the
mirror image of the Data Provider, with the entire Big Data framework
appearing like a Data Provider to the Data Consumer. The activities
associated with the Data Consumer role include the following:

* Search and Retrieve,
* Download,
* Analyze Locally,
* Reporting,
* Visualization, and
* Data to Use for Their Own Processes.

The interface for the data consumer must be able to describe the
consuming services and how they retrieve information or leverage data
consumers.

### Big Data Application Interface Provider Requirements {#sec:data-application-requirements}

The Big Data Application Provider role executes a specific set of
operations along the data life cycle to meet the requirements
established by the System Orchestrator, as well as meeting security and
privacy requirements. The Big Data Application Provider is the
architecture component that encapsulates the business logic and
functionality to be executed by the architecture. The interfaces to
describe Big Data applications include interfaces for the various
subcomponents including collections, preparation/curation, analytics,
visualization, and access. Some of the interfaces used in these
subcomponents can be reused from other interfaces, which are introduced
in other sections of this document. Where appropriate,
application-specific interfaces will be identified and examples provided
with a focus on use cases as identified in the *NBDIF: Volume 3, Use
Cases and General Requirements*.

#### Collection

In general, the collection activity of the Big Data Application Provider
handles the interface with the Data Provider. This may be a general
service, such as a file server or web server configured by the System
Orchestrator to accept or perform specific collections of data, or it
may be an application-specific service designed to pull data or receive
pushes of data from the Data Provider. Since this activity is receiving
data at a minimum, it must store/buffer the received data until it is
persisted through the Big Data Framework Provider. This persistence need
not be to physical media but may simply be to an in-memory queue or
other service provided by the processing frameworks of the Big Data
Framework Provider. The collection activity is likely where the
extraction portion of the Extract, Transform, Load (ETL)/Extract, Load,
Transform (ELT) cycle is performed. At the initial collection stage,
sets of data (e.g., data records) of similar structure are collected
(and combined), resulting in uniform security, policy, and other
considerations. Initial metadata is created (e.g., subjects with keys
are identified) to facilitate subsequent aggregation or look-up methods.

#### Preparation

The preparation activity is where the transformation portion of the
ETL/ELT cycle is likely performed, although analytics activity will also
likely perform advanced parts of the transformation. Tasks performed by
this activity could include data validation (e.g., checksums/hashes,
format checks), cleaning (e.g., eliminating bad records/fields),
outlier removal, standardization, reformatting, or encapsulating. This
activity is also where source data will frequently be persisted to
archive storage in the Big Data Framework Provider and provenance data
will be verified or attached/associated. Verification or attachment may
include optimization of data through manipulations (e.g., deduplication)
and indexing to optimize the analytics process. This activity may also
aggregate data from different Data Providers, leveraging metadata keys
to create an expanded and enhanced data set.

#### Analytics

The analytics activity of the Big Data Application Provider includes the
encoding of the low-level business logic of the Big Data system (with
higher-level business process logic being encoded by the System
Orchestrator). The activity implements the techniques to extract
knowledge from the data based on the requirements of the vertical
application. The requirements specify the data processing algorithms to
produce new insights that will address the technical goal. The analytics
activity will leverage the processing frameworks to implement the
associated logic. This typically involves the activity providing
software that implements the analytic logic to the batch and/or
streaming elements of the processing framework for execution. The
messaging/communication framework of the Big Data Framework Provider may
be used to pass data or control functions to the application logic
running in the processing frameworks. The analytic logic may be broken
up into multiple modules to be executed by the processing frameworks
which communicate, through the messaging/communication framework, with
each other and other functions instantiated by the Big Data Application
Provider.

#### Visualization

The visualization activity of the Big Data Application Provider prepares
elements of the processed data and the output of the analytic activity
for presentation to the Data Consumer. The objective of this activity is
to format and present data in such a way as to optimally communicate
meaning and knowledge. The visualization preparation may involve
producing a text-based report or rendering the analytic results as some
form of graphic. The resulting output may be a static visualization and
may simply be stored through the Big Data Framework Provider for later
access. However, the visualization activity frequently interacts with
the access activity, the analytics activity, and the Big Data Framework
Provider (processing and platform) to provide interactive visualization
of the data to the Data Consumer based on parameters provided to the
access activity by the Data Consumer. The visualization activity may be
completely application-implemented, leverage one or more application
libraries, or may use specialized visualization processing frameworks
within the Big Data Framework Provider.

#### Access

The access activity within the Big Data Application Provider is focused
on the communication/interaction with the Data Consumer. Like the
collection activity, the access activity may be a generic service such
as a web server or application server that is configured by the System
Orchestrator to handle specific requests from the Data Consumer. This
activity would interface with the visualization and analytic activities
to respond to requests from the Data Consumer (who may be a person) and
uses the processing and platform frameworks to retrieve data to respond
to Data Consumer requests. In addition, the access activity confirms
that descriptive and administrative metadata and metadata schemes are
captured and maintained for access by the Data Consumer and as data is
transferred to the Data Consumer. The interface with the Data Consumer
may be synchronous or asynchronous in nature and may use a pull or push
paradigm for data transfer.

### Big Data Provider Framework Interface Requirements {#sec:provider-requirements}

Data for Big Data applications are delivered through data providers.
They can be either local providers, data contributed by a user, or
distributed data providers, data on the Internet. This interface must be
able to provide the following functionality:

* Interfaces to files,
* Interfaces to virtual data directories,
* Interfaces to data streams, and
* Interfaces to data filters.

#### Infrastructures Interface Requirements

This Big Data Framework Provider element provides all the resources
necessary to host/run the activities of the other components of the Big
Data system. Typically, these resources consist of some combination of
physical resources, which may host/support similar virtual resources.
The NBDRA needs interfaces that can be used to deal with the underlying
infrastructure to address networking, computing, and storage.

#### Platforms Interface Requirements

As part of the NBDRA platforms, interfaces are needed that can address
platform needs and services for data organization, data distribution,
indexed storage, and file systems.

#### Processing Interface Requirements

The processing frameworks for Big Data provide the necessary
infrastructure software to support implementation of applications that
can deal with the volume, velocity, variety, and variability of data.
Processing frameworks define how the computation and processing of the
data is organized. Big Data applications rely on various platforms and
technologies to meet the challenges of scalable data analytics and
operation. A requirement is the ability to interface easily with
computing services that offer specific analytics services, batch
processing capabilities, interactive analysis, and data streaming.

#### Crosscutting Interface Requirements

Several crosscutting interface requirements within the Big Data
Framework Provider include messaging, communication, and resource
management. Often these services may be hidden from explicit interface
use as they are part of larger systems that expose higher-level
functionality through their interfaces. However, such interfaces may
also be exposed on a lower level in case finer-grained control is
needed. The need for such crosscutting interface requirements will be
extracted from the *NBDIF: Volume 3, Use Cases and General Requirements*
document.

#### Messaging/Communications Frameworks

Messaging and communications frameworks have their roots in the High
Performance Computing environments long popular in the scientific and
research communities. Messaging/Communications Frameworks were developed
to provide application programming interfaces (APIs) for the reliable
queuing, transmission, and receipt of data.

#### Resource Management Framework

As Big Data systems have evolved and become more complex, and as
businesses work to leverage limited computation and storage resources to
address a broader range of applications and business challenges, the
requirement to effectively manage those resources has grown
significantly. While tools for resource management and *elastic
computing* have expanded and matured in response to the needs of cloud
providers and virtualization technologies, Big Data introduces unique
requirements for these tools. However, Big Data frameworks tend to fall
more into a distributed computing paradigm, which presents additional
challenges.

### Big Data Application Provider to Big Data Framework Provider Interface {#sec:app-provider-requirements}

The Big Data Framework Provider typically consists of one or more
hierarchically organized instances of the components in the NBDRA IT
value chain (@fig:arch). There is no requirement that all instances at a
given level in the hierarchy be of the same technology. In fact, most
Big Data implementations are hybrids that combine multiple technology
approaches to provide flexibility or meet the complete range of
requirements, which are driven from the Big Data Application Provider.

