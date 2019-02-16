//
//  MyFairSemaphore2.hpp
//  Lab
//
//  Created by Ciprian on 10/12/18.
//  Copyright © 2018 Ciprian. All rights reserved.
//

#ifndef MyFairSemaphore2_hpp
#define MyFairSemaphore2_hpp

#include <stdio.h>
#include <queue>

#include "MySemphore.hpp"

using base_type = MySemaphore;

class MyFairThreadContext : public MyThreadContextBase
{
public:
    std::mutex m_personalMutex;
};

class MyFairSemaphore2 : public base_type
{
public:
    
    // Rewrite the signal and wait functions
    void signal(MyThreadContextBase* context = nullptr)
    {
        std::lock_guard<std::mutex> guardLock(m_queueSyncMutex);
        
        m_counter++; // Another slot is free now.
        
        // Take the next free item from list and run it.
        if (m_waitingQueue.empty() == false)
        {
            std::mutex* vc = m_waitingQueue.front();
            m_waitingQueue.pop();
            
            // Let another thread continue
            vc->unlock();
        }
    }
    
    void wait(MyThreadContextBase* context = nullptr)
    {
        MyFairThreadContext* fairContext = static_cast<MyFairThreadContext*>(context);
        std::mutex* mutexToUse = &(fairContext->m_personalMutex);
        
        std::unique_lock<std::mutex> guardLock(m_queueSyncMutex);
        
        // If no free slots available, lock me and wait for a signal (from a different working thread).
        if (m_counter == 0)
        {
            // Add me to the queue
            m_waitingQueue.push(mutexToUse);
            mutexToUse->lock();
        }
        
        m_counter--;
    }
    
    bool try_wait(MyThreadContextBase* context = nullptr)
    {
        return base_type::try_wait(context);
    }
    
private:
    
    // Keep a pointers to all condition variables
    // TODO: use smart pointers ?
    std::queue<std::mutex*> m_waitingQueue;
    
    std::mutex m_queueSyncMutex;
    
};

#endif /* MyFairSemaphore2_hpp */
