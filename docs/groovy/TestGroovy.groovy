class TestGroovy {
    // This is a test Groovy class
    String name
    
    TestGroovy(String name) {
        this.name = name
    }
    
    String getName() {
        return name
    }
    
    void setName(String name) {
        this.name = name
    }
    
    String toString() {
        return "TestGroovy[name='${name}']"
    }
} 