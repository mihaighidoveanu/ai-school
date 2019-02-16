//
//  MySemphore.hpp
//  Lab
//
//  Created by Ciprian on 10/10/18.
//  Copyright © 2018 Ciprian. All rights reserved.
//

#ifndef MySemphore_hpp
#define MySemphore_hpp


#include <mutex>
#include <condition_variable>
#include <assert.h>

#define SANITY_CHECKS_ENABLED // Enable only on debug builds !

// This is a custom thread context that you may send to the sempahore to let him knows tricky details about you
class MyThreadContextBase
{
    // Empty by default
};


class MySemaphore
{
    
public:
    
    // Let the user init the value of the semaphore with a desired value
    void init(const int _count)
    {
        m_counter = _count;
        m_initialCounter = m_counter;
    }
    
    // Notify one waiting to continue
    virtual void signal(MyThreadContextBase* context = nullptr)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        _increaseCounter();
        m_condition.notify_one(); // Signal one that is waiting
    }
    
    // Wait until a new signal is emited
    virtual void wait(MyThreadContextBase* context = nullptr)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        
        while(!m_counter)
            m_condition.wait(lock);
        
        // We can replace the two lines above with m_condition.wait(lock, [&]{return m_counter > 0;})
        
        _decreaseCounter();
        
    }
    
    // non blocking wait. Return true if semaphore has been taken, false otherwise
    virtual bool try_wait(MyThreadContextBase* context = nullptr)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_counter)
        {
            _decreaseCounter();
            _counterSanityCheck();
            return true;
        }
        
        return false;
    }
    
protected:
    unsigned long m_counter = 0; // HOw many signals (free slots) are available in this moment. Locked by default (0)
    unsigned long m_initialCounter = 0; // This is equal to m_counter initially set by user. Used for sanity check that we don't ever make m_counter > m_initialCounter
    
#ifdef SANITY_CHECKS_ENABLED
    void _counterSanityCheck()
    {
        assert(m_counter >=0 && m_counter <= m_initialCounter);
    }
#endif
    
    inline void _increaseCounter()
    {
        m_counter++;
        
#ifdef SANITY_CHECKS_ENABLED
        _counterSanityCheck();
#endif
    }
    
    inline void _decreaseCounter()
    {
        m_counter--;
        
#ifdef SANITY_CHECKS_ENABLED
        _counterSanityCheck();
#endif
    }
    
private:
    std::mutex m_mutex;
    std::condition_variable m_condition;
};


#endif /* MySemphore_hpp */
