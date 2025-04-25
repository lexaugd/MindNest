# Object Wrapper System

## Overview

The Object Wrapper System provides a consistent interface for accessing different types of objects throughout the application. It allows for a uniform approach to data access regardless of the underlying data structure, simplifying business logic and making code more maintainable.

## Architecture Diagram

```
┌───────────────────┐     ┌───────────────────┐
│                   │     │                   │
│   Client Code     │────▶│  ObjectWrapper    │
│                   │     │     Interface     │
└───────────────────┘     └─────────┬─────────┘
                                    │
                                    ▼
                          ┌───────────────────┐
                          │                   │
                          │    WrapperFactory │
                          │                   │
                          └─────────┬─────────┘
                                    │
           ┌────────────┬───────────┼───────────┬────────────┐
           │            │           │           │            │
           ▼            ▼           ▼           ▼            ▼
  ┌─────────────┐ ┌───────────┐ ┌─────────┐ ┌─────────┐ ┌───────────┐
  │             │ │           │ │         │ │         │ │           │
  │ MapWrapper  │ │ListWrapper│ │Infonode │ │Property │ │  Other    │
  │             │ │           │ │Wrapper  │ │Wrapper  │ │ Wrappers  │
  └─────────────┘ └───────────┘ └─────────┘ └─────────┘ └───────────┘
```

## Core Wrapper Types

### 1. ObjectWrapper

The base wrapper that provides common functionality for all wrapped objects:

```java
public interface ObjectWrapper {
  // Presence check
  boolean isPresent();
  
  // Get underlying object
  Object unwrap();
  
  // Property access
  Set<String> getKeys();
  ObjectWrapper get(String key);
  
  // Type conversion
  String asString();
  Integer asInteger();
  BigDecimal asDecimal();
  Boolean asBoolean();
  Date asDate();
  ListWrapper asList();
  MapWrapper asMap();
  
  // Default values
  String asString(String defaultValue);
  Integer asInteger(Integer defaultValue);
  // ...
}
```

### 2. InfonodeWrapper

Wraps objects from the CD (Customer Data) system:

```java
class InfonodeWrapper implements ObjectWrapper {
  private Infonode node;
  
  InfonodeWrapper(Infonode node) {
    this.node = node;
  }
  
  boolean isPresent() {
    return node != null;
  }
  
  Object unwrap() {
    return node;
  }
  
  // Retrieve a property by name
  ObjectWrapper get(String key) {
    if (node == null) return new Empty();
    try {
      Object value = node.get(Attribute.getInstance(key));
      return WrapperFactory.wrap(value);
    } catch (Exception e) {
      return new Empty();
    }
  }
  
  // Get all keys available on this node
  Set<String> getKeys() {
    if (node == null) return Collections.emptySet();
    return Arrays.stream(node.getBoundAttributeNames())
      .map(name -> name.toString())
      .collect(Collectors.toSet());
  }
  
  // Type conversion methods
  // ...
}
```

### 3. MapWrapper / MapEntryWrapper

Wraps Map objects and map entries for uniform access:

```java
class MapWrapper implements ObjectWrapper {
  private Map<String, Object> map;
  
  MapWrapper(Map<String, Object> map) {
    this.map = map != null ? map : Collections.emptyMap();
  }
  
  boolean isPresent() {
    return map != null && !map.isEmpty();
  }
  
  Object unwrap() {
    return map;
  }
  
  ObjectWrapper get(String key) {
    if (map == null) return new Empty();
    Object value = map.get(key);
    return WrapperFactory.wrap(value);
  }
  
  Set<String> getKeys() {
    if (map == null) return Collections.emptySet();
    return map.keySet();
  }
  
  // Other implementation methods
  // ...
}
```

### 4. ListWrapper

Wraps List objects for uniform iteration and access:

```java
class ListWrapper implements ObjectWrapper {
  private List<ObjectWrapper> items;
  
  ListWrapper(List<?> list) {
    if (list == null) {
      this.items = Collections.emptyList();
    } else {
      this.items = list.stream()
        .map(WrapperFactory::wrap)
        .collect(Collectors.toList());
    }
  }
  
  boolean isPresent() {
    return items != null && !items.isEmpty();
  }
  
  Object unwrap() {
    return items.stream()
      .map(ObjectWrapper::unwrap)
      .collect(Collectors.toList());
  }
  
  int size() {
    return items.size();
  }
  
  ObjectWrapper get(int index) {
    if (index < 0 || index >= items.size()) {
      return new Empty();
    }
    return items.get(index);
  }
  
  void forEach(Consumer<ObjectWrapper> consumer) {
    items.forEach(consumer);
  }
  
  // Other implementation methods
  // ...
}
```

### 5. PropertyWrapper

Wraps Java Bean properties for uniform access:

```java
class PropertyWrapper implements ObjectWrapper {
  private Object bean;
  private PropertyDescriptor descriptor;
  
  PropertyWrapper(Object bean, PropertyDescriptor descriptor) {
    this.bean = bean;
    this.descriptor = descriptor;
  }
  
  boolean isPresent() {
    return bean != null && descriptor != null && descriptor.getReadMethod() != null;
  }
  
  Object unwrap() {
    if (!isPresent()) return null;
    try {
      Method readMethod = descriptor.getReadMethod();
      return readMethod.invoke(bean);
    } catch (Exception e) {
      return null;
    }
  }
  
  // Other implementation methods
  // ...
}
```

### 6. Function Implementations

Some wrapper implementations are used specifically for function evaluation:

```java
class Constant implements ObjectWrapper {
  private Object value;
  
  Constant(Object value) {
    this.value = value;
  }
  
  boolean isPresent() {
    return value != null;
  }
  
  Object unwrap() {
    return value;
  }
  
  // Type conversion methods
  // ...
}

class Empty implements ObjectWrapper {
  boolean isPresent() {
    return false;
  }
  
  Object unwrap() {
    return null;
  }
  
  // Type conversion methods that return defaults
  String asString() {
    return "";
  }
  
  Integer asInteger() {
    return null;
  }
  
  // ...
}
```

## WrapperFactory

The WrapperFactory is responsible for creating the appropriate wrapper based on the type of object:

```java
public class WrapperFactory {
  public static ObjectWrapper wrap(Object object) {
    if (object == null) {
      return new Empty();
    } else if (object instanceof Infonode) {
      return new InfonodeWrapper((Infonode) object);
    } else if (object instanceof Map) {
      return new MapWrapper((Map<String, Object>) object);
    } else if (object instanceof List) {
      return new ListWrapper((List<?>) object);
    } else if (object instanceof String || object instanceof Number || object instanceof Boolean) {
      return new Constant(object);
    } else {
      // For regular Java beans or other objects
      return new BeanWrapper(object);
    }
  }
}
```

## Special Purpose Wrappers

### 1. ExtensionWrapper (CD System)

Provides access to CD system extensions:

```java
class ExtensionWrapper implements ObjectWrapper {
  private Infonode node;
  private Attribute attribute;
  private ExtensionUtil extensionUtil;
  private Integer extensionId;
  private String extensionName;
  
  ExtensionWrapper(Infonode node, Attribute attribute, ExtensionUtil extensionUtil) {
    this.node = node;
    this.attribute = attribute;
    this.extensionUtil = extensionUtil;
    this.extensionId = extensionUtil.getExtensionId(node);
    this.extensionName = CommitOrder.getDetailExtensionMap().get(attribute);
  }
  
  Object unwrap() {
    if (!isPresent()) return null;
    return extensionUtil.getExtensionValue(extensionId, extensionName);
  }
  
  // Set an extension value
  void setValue(Object value) {
    if (isPresent()) {
      extensionUtil.setExtensionValue(extensionId, extensionName, value);
    }
  }
  
  // Other implementation methods
  // ...
}
```

### 2. KeyWrapper (CD System)

Wraps CD system keys:

```java
class KeyWrapper implements ObjectWrapper {
  private Infokey key;
  
  KeyWrapper(Infokey key) {
    this.key = key;
  }
  
  boolean isPresent() {
    return key != null;
  }
  
  Object unwrap() {
    return key;
  }
  
  String getKeyType() {
    if (!isPresent()) return "";
    return key.getClass().getSimpleName();
  }
  
  // Other implementation methods
  // ...
}
```

## Integration with ResourceDrivers

Object wrappers are frequently used with resources provided by ResourceDrivers, creating a seamless interface between the resource management and data access layers.

### Example with Database Resources

```groovy
void processOrder(ServiceContext context, ProcessLog log, Action action, Result result) {
  // Get the database resource via ResourceDriver
  DatabaseResourceDriver dbDriver = context.get("databaseDriver");
  DatabaseResource db = dbDriver.prepare(context, log, action, result);
  
  try {
    // Retrieve order data and wrap it
    Map<String, Object> orderData = db.findOrderById("ORD12345");
    ObjectWrapper order = WrapperFactory.wrap(orderData);
    
    // Access order properties with the wrapper
    String customerId = order.get("customerId").asString();
    ObjectWrapper items = order.get("items").asList();
    
    // Process each item
    items.forEach(item -> {
      String sku = item.get("sku").asString();
      int quantity = item.get("quantity").asInteger(0);
      BigDecimal price = item.get("price").asDecimal(BigDecimal.ZERO);
      
      // Business logic...
    });
    
    // Successful completion
    dbDriver.finished(context, log, action, result, db);
  } finally {
    // Cleanup
    dbDriver.cleanup(context, log, action, db);
  }
}
```

### Example with CD System Resources

```groovy
void processShipment(ServiceContext context, ProcessLog log, Action action, Result result) {
  // Get CD resource via ResourceDriver
  Cd1ResourceDriver cd1Driver = context.get("cd1Driver");
  Cd1Resource cd1 = cd1Driver.prepare(context, log, action, result);
  
  try {
    // Find shipment in CD system
    Cd1ShipmentInput shipmentInput = cd1.findShipment([trackingNo: "1Z9999999"]);
    
    // Wrap the shipment for easy access
    ObjectWrapper shipment = WrapperFactory.wrap(shipmentInput.getShipment());
    
    // Access shipment properties
    String status = shipment.get("status").asString();
    Date deliveryDate = shipment.get("deliveryDate").asDate();
    
    // Process shipment packages
    ObjectWrapper packages = shipment.get("packages").asList();
    packages.forEach(pkg -> {
      String packageId = pkg.get("id").asString();
      BigDecimal weight = pkg.get("weight").asDecimal();
      
      // Business logic...
    });
    
    // Successful completion
    cd1Driver.finished(context, log, action, result, cd1);
  } finally {
    // Cleanup
    cd1Driver.cleanup(context, log, action, cd1);
  }
}
```

## Usage Patterns

### 1. Unified Property Access

```groovy
// Access the same way regardless of underlying object type
ObjectWrapper wrapper = WrapperFactory.wrap(someObject);
String name = wrapper.get("name").asString();
int age = wrapper.get("age").asInteger(0);  // Default to 0 if missing
```

### 2. Safe Navigation

```groovy
// No need to check for null at each step
ObjectWrapper order = getOrder();  // This could return an Empty wrapper if no order exists
String customerName = order.get("customer").get("name").asString();  // Safe even if customer is null
```

### 3. Type Conversion

```groovy
// Automatic type conversion
int quantity = wrapper.get("quantity").asInteger();
BigDecimal price = wrapper.get("price").asDecimal();
Date shipDate = wrapper.get("shipDate").asDate();
boolean isPriority = wrapper.get("priority").asBoolean();
```

### 4. Collection Processing

```groovy
// Iterate over wrapped collections
ListWrapper items = order.get("items").asList();
items.forEach(item -> {
  // Process each item with type safety
  String sku = item.get("sku").asString();
  int qty = item.get("quantity").asInteger(0);
  BigDecimal price = item.get("price").asDecimal(BigDecimal.ZERO);
  BigDecimal total = price.multiply(new BigDecimal(qty));
  
  // Business logic...
});
```

### 5. Null-Safe Transformations

```groovy
// Transform data with null safety
List<String> skus = new ArrayList<>();
order.get("items").asList().forEach(item -> {
  String sku = item.get("sku").asString();
  if (!sku.isEmpty()) {
    skus.add(sku);
  }
});
```

## Implementation Patterns

### 1. Chain of Responsibility

Wrappers often use a chain of responsibility pattern to delegate operations:

```groovy
// Base class delegates to specific implementations
public abstract class AbstractWrapper implements ObjectWrapper {
  @Override
  public ObjectWrapper get(String key) {
    Object value = getPropertyValue(key);
    return WrapperFactory.wrap(value);
  }
  
  // Subclasses implement this method
  protected abstract Object getPropertyValue(String key);
}
```

### 2. Null Object Pattern

The Empty wrapper is an implementation of the Null Object pattern:

```groovy
class Empty implements ObjectWrapper {
  // Always indicate not present
  boolean isPresent() {
    return false;
  }
  
  // Return safe default values
  String asString() { return ""; }
  Integer asInteger() { return null; }
  BigDecimal asDecimal() { return null; }
  Boolean asBoolean() { return false; }
  Date asDate() { return null; }
  ListWrapper asList() { return new ListWrapper(Collections.emptyList()); }
  MapWrapper asMap() { return new MapWrapper(Collections.emptyMap()); }
  
  // Other methods...
}
```

### 3. Factory Method Pattern

The WrapperFactory uses the Factory Method pattern to create appropriate wrappers:

```groovy
public class WrapperFactory {
  public static ObjectWrapper wrap(Object object) {
    if (object == null) {
      return new Empty();
    } else if (object instanceof ObjectWrapper) {
      return (ObjectWrapper) object;
    } else if (object instanceof Infonode) {
      return new InfonodeWrapper((Infonode) object);
    }
    // Other type checks...
  }
}
```

## Best Practices

### 1. Use WrapperFactory

Always create wrappers using the WrapperFactory rather than direct instantiation to ensure the correct wrapper type is used.

```groovy
// Good
ObjectWrapper wrapper = WrapperFactory.wrap(someObject);

// Avoid
ObjectWrapper wrapper = new MapWrapper((Map<String, Object>) someObject);  // Could throw ClassCastException
```

### 2. Handle Defaults Appropriately

Use the overloaded conversion methods with defaults for critical values:

```groovy
// Good
int quantity = item.get("quantity").asInteger(0);  // Default to 0 if missing
BigDecimal price = item.get("price").asDecimal(BigDecimal.ZERO);  // Default to 0 if missing

// Avoid
int quantity = item.get("quantity").asInteger();  // Could be null
BigDecimal price = item.get("price").asDecimal();  // Could be null
```

### 3. Chain Property Access Safely

Take advantage of the null-safe property chaining:

```groovy
// Good
String city = customer.get("address").get("city").asString();

// Avoid
if (customer.isPresent()) {
  ObjectWrapper address = customer.get("address");
  if (address.isPresent()) {
    city = address.get("city").asString();
  }
}
```

### 4. Return Empty Collections Instead of Null

When implementing wrappers, always return empty collections rather than null:

```groovy
// Good
Set<String> getKeys() {
  if (map == null) return Collections.emptySet();
  return map.keySet();
}

// Avoid
Set<String> getKeys() {
  if (map == null) return null;  // Forces null check on caller
  return map.keySet();
}
```

## Benefits

1. **Unified Interface**: Provides a consistent way to access different types of objects.
2. **Null Safety**: Eliminates many null checks through the `isPresent()` pattern.
3. **Type Conversion**: Handles type conversion between different data representations.
4. **Immutability**: Wrappers are typically immutable, making them thread-safe.
5. **Separation of Concerns**: Decouples business logic from the details of data access.
6. **Error Resilience**: Gracefully handles missing properties and type conversion errors.

## Related Documentation

For more information about how Object Wrappers integrate with other components:

- [Core Architecture](01_Core_Architecture.md)
- [ResourceDriver Pattern](02_ResourceDriver_Pattern.md)
- [CD System Integration](04_CD_System_Integration.md)
- [Value System](07_Value_System.md) 